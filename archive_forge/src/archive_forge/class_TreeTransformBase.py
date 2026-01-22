import errno
import os
import posixpath
import tempfile
import time
from stat import S_IEXEC, S_ISREG
from dulwich.index import blob_from_path_and_stat, commit_tree
from dulwich.objects import Blob
from .. import annotate, conflicts, errors, multiparent, osutils
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..i18n import gettext
from ..mutabletree import MutableTree
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..tree import InterTree, TreeChange
from .mapping import (decode_git_path, encode_git_path, mode_is_executable,
from .tree import GitTree, GitTreeDirectory, GitTreeFile, GitTreeSymlink
class TreeTransformBase(TreeTransform):
    """The base class for TreeTransform and its kin."""

    def __init__(self, tree, pb=None, case_sensitive=True):
        """Constructor.

        :param tree: The tree that will be transformed, but not necessarily
            the output tree.
        :param pb: ignored
        :param case_sensitive: If True, the target of the transform is
            case sensitive, not just case preserving.
        """
        super().__init__(tree, pb=pb)
        self._observed_sha1s = {}
        self._versioned = set()
        self.root = self.trans_id_tree_path('')
        self._case_sensitive_target = case_sensitive
        self._symlink_target = {}

    @property
    def mapping(self):
        return self._tree.mapping

    def finalize(self):
        """Release the working tree lock, if held.

        This is required if apply has not been invoked, but can be invoked
        even after apply.
        """
        if self._tree is None:
            return
        for hook in MutableTree.hooks['post_transform']:
            hook(self._tree, self)
        self._tree.unlock()
        self._tree = None

    def create_path(self, name, parent):
        """Assign a transaction id to a new path"""
        trans_id = self.assign_id()
        unique_add(self._new_name, trans_id, name)
        unique_add(self._new_parent, trans_id, parent)
        return trans_id

    def adjust_root_path(self, name, parent):
        """Emulate moving the root by moving all children, instead.
        """

    def fixup_new_roots(self):
        """Reinterpret requests to change the root directory

        Instead of creating a root directory, or moving an existing directory,
        all the attributes and children of the new root are applied to the
        existing root directory.

        This means that the old root trans-id becomes obsolete, so it is
        recommended only to invoke this after the root trans-id has become
        irrelevant.

        """
        new_roots = [k for k, v in self._new_parent.items() if v == ROOT_PARENT]
        if len(new_roots) < 1:
            return
        if len(new_roots) != 1:
            raise ValueError('A tree cannot have two roots!')
        old_new_root = new_roots[0]
        if old_new_root in self._versioned:
            self.cancel_versioning(old_new_root)
        else:
            self.unversion_file(old_new_root)
        list(self.iter_tree_children(old_new_root))
        for child in self.by_parent().get(old_new_root, []):
            self.adjust_path(self.final_name(child), self.root, child)
        if old_new_root in self._new_contents:
            self.cancel_creation(old_new_root)
        else:
            self.delete_contents(old_new_root)
        if self.root in self._removed_contents:
            self.cancel_deletion(self.root)
        del self._new_parent[old_new_root]
        del self._new_name[old_new_root]

    def trans_id_file_id(self, file_id):
        """Determine or set the transaction id associated with a file ID.
        A new id is only created for file_ids that were never present.  If
        a transaction has been unversioned, it is deliberately still returned.
        (this will likely lead to an unversioned parent conflict.)
        """
        if file_id is None:
            raise ValueError('None is not a valid file id')
        path = self.mapping.parse_file_id(file_id)
        return self.trans_id_tree_path(path)

    def version_file(self, trans_id, file_id=None):
        """Schedule a file to become versioned."""
        if trans_id in self._versioned:
            raise errors.DuplicateKey(key=trans_id)
        self._versioned.add(trans_id)

    def cancel_versioning(self, trans_id):
        """Undo a previous versioning of a file"""
        raise NotImplementedError(self.cancel_versioning)

    def new_paths(self, filesystem_only=False):
        """Determine the paths of all new and changed files.

        :param filesystem_only: if True, only calculate values for files
            that require renames or execute bit changes.
        """
        new_ids = set()
        if filesystem_only:
            stale_ids = self._needs_rename.difference(self._new_name)
            stale_ids.difference_update(self._new_parent)
            stale_ids.difference_update(self._new_contents)
            stale_ids.difference_update(self._versioned)
            needs_rename = self._needs_rename.difference(stale_ids)
            id_sets = (needs_rename, self._new_executability)
        else:
            id_sets = (self._new_name, self._new_parent, self._new_contents, self._versioned, self._new_executability)
        for id_set in id_sets:
            new_ids.update(id_set)
        return sorted(FinalPaths(self).get_paths(new_ids))

    def final_is_versioned(self, trans_id):
        if trans_id in self._versioned:
            return True
        if trans_id in self._removed_id:
            return False
        orig_path = self.tree_path(trans_id)
        if orig_path is None:
            return False
        return self._tree.is_versioned(orig_path)

    def find_raw_conflicts(self):
        """Find any violations of inventory or filesystem invariants"""
        if self._done is True:
            raise ReusingTransform()
        conflicts = []
        self._add_tree_children()
        by_parent = self.by_parent()
        conflicts.extend(self._parent_loops())
        conflicts.extend(self._duplicate_entries(by_parent))
        conflicts.extend(self._parent_type_conflicts(by_parent))
        conflicts.extend(self._improper_versioning())
        conflicts.extend(self._executability_conflicts())
        conflicts.extend(self._overwrite_conflicts())
        return conflicts

    def _check_malformed(self):
        conflicts = self.find_raw_conflicts()
        if len(conflicts) != 0:
            raise MalformedTransform(conflicts=conflicts)

    def _add_tree_children(self):
        """Add all the children of all active parents to the known paths.

        Active parents are those which gain children, and those which are
        removed.  This is a necessary first step in detecting conflicts.
        """
        parents = list(self.by_parent())
        parents.extend([t for t in self._removed_contents if self.tree_kind(t) == 'directory'])
        for trans_id in self._removed_id:
            path = self.tree_path(trans_id)
            if path is not None:
                try:
                    if self._tree.stored_kind(path) == 'directory':
                        parents.append(trans_id)
                except _mod_transport.NoSuchFile:
                    pass
            elif self.tree_kind(trans_id) == 'directory':
                parents.append(trans_id)
        for parent_id in parents:
            list(self.iter_tree_children(parent_id))

    def _has_named_child(self, name, parent_id, known_children):
        """Does a parent already have a name child.

        :param name: The searched for name.

        :param parent_id: The parent for which the check is made.

        :param known_children: The already known children. This should have
            been recently obtained from `self.by_parent.get(parent_id)`
            (or will be if None is passed).
        """
        if known_children is None:
            known_children = self.by_parent().get(parent_id, [])
        for child in known_children:
            if self.final_name(child) == name:
                return True
        parent_path = self._tree_id_paths.get(parent_id, None)
        if parent_path is None:
            return False
        child_path = joinpath(parent_path, name)
        child_id = self._tree_path_ids.get(child_path, None)
        if child_id is None:
            return osutils.lexists(self._tree.abspath(child_path))
        else:
            raise AssertionError('child_id is missing: %s, %s, %s' % (name, parent_id, child_id))

    def _available_backup_name(self, name, target_id):
        """Find an available backup name.

        :param name: The basename of the file.

        :param target_id: The directory trans_id where the backup should
            be placed.
        """
        known_children = self.by_parent().get(target_id, [])
        return osutils.available_backup_name(name, lambda base: self._has_named_child(base, target_id, known_children))

    def _parent_loops(self):
        """No entry should be its own ancestor"""
        for trans_id in self._new_parent:
            seen = set()
            parent_id = trans_id
            while parent_id != ROOT_PARENT:
                seen.add(parent_id)
                try:
                    parent_id = self.final_parent(parent_id)
                except KeyError:
                    break
                if parent_id == trans_id:
                    yield ('parent loop', trans_id)
                if parent_id in seen:
                    break

    def _improper_versioning(self):
        """Cannot version a file with no contents, or a bad type.

        However, existing entries with no contents are okay.
        """
        for trans_id in self._versioned:
            kind = self.final_kind(trans_id)
            if kind == 'symlink' and (not self._tree.supports_symlinks()):
                continue
            if kind is None:
                yield ('versioning no contents', trans_id)
                continue
            if not self._tree.versionable_kind(kind):
                yield ('versioning bad kind', trans_id, kind)

    def _executability_conflicts(self):
        """Check for bad executability changes.

        Only versioned files may have their executability set, because
        1. only versioned entries can have executability under windows
        2. only files can be executable.  (The execute bit on a directory
           does not indicate searchability)
        """
        for trans_id in self._new_executability:
            if not self.final_is_versioned(trans_id):
                yield ('unversioned executability', trans_id)
            elif self.final_kind(trans_id) != 'file':
                yield ('non-file executability', trans_id)

    def _overwrite_conflicts(self):
        """Check for overwrites (not permitted on Win32)"""
        for trans_id in self._new_contents:
            if self.tree_kind(trans_id) is None:
                continue
            if trans_id not in self._removed_contents:
                yield ('overwrite', trans_id, self.final_name(trans_id))

    def _duplicate_entries(self, by_parent):
        """No directory may have two entries with the same name."""
        if (self._new_name, self._new_parent) == ({}, {}):
            return
        for children in by_parent.values():
            name_ids = []
            for child_tid in children:
                name = self.final_name(child_tid)
                if name is not None:
                    if not self._case_sensitive_target:
                        name = name.lower()
                    name_ids.append((name, child_tid))
            name_ids.sort()
            last_name = None
            last_trans_id = None
            for name, trans_id in name_ids:
                kind = self.final_kind(trans_id)
                if kind is None and (not self.final_is_versioned(trans_id)):
                    continue
                if name == last_name:
                    yield ('duplicate', last_trans_id, trans_id, name)
                last_name = name
                last_trans_id = trans_id

    def _parent_type_conflicts(self, by_parent):
        """Children must have a directory parent"""
        for parent_id, children in by_parent.items():
            if parent_id == ROOT_PARENT:
                continue
            no_children = True
            for child_id in children:
                if self.final_kind(child_id) is not None:
                    no_children = False
                    break
            if no_children:
                continue
            kind = self.final_kind(parent_id)
            if kind is None:
                yield ('missing parent', parent_id)
            elif kind != 'directory':
                yield ('non-directory parent', parent_id)

    def _set_executability(self, path, trans_id):
        """Set the executability of versioned files """
        if self._tree._supports_executable():
            new_executability = self._new_executability[trans_id]
            abspath = self._tree.abspath(path)
            current_mode = os.stat(abspath).st_mode
            if new_executability:
                umask = os.umask(0)
                os.umask(umask)
                to_mode = current_mode | 64 & ~umask
                if current_mode & 4:
                    to_mode |= 1 & ~umask
                if current_mode & 32:
                    to_mode |= 8 & ~umask
            else:
                to_mode = current_mode & ~73
            osutils.chmod_if_possible(abspath, to_mode)

    def _new_entry(self, name, parent_id, file_id):
        """Helper function to create a new filesystem entry."""
        trans_id = self.create_path(name, parent_id)
        if file_id is not None:
            self.version_file(trans_id, file_id=file_id)
        return trans_id

    def new_file(self, name, parent_id, contents, file_id=None, executable=None, sha1=None):
        """Convenience method to create files.

        name is the name of the file to create.
        parent_id is the transaction id of the parent directory of the file.
        contents is an iterator of bytestrings, which will be used to produce
        the file.
        :param file_id: The inventory ID of the file, if it is to be versioned.
        :param executable: Only valid when a file_id has been supplied.
        """
        trans_id = self._new_entry(name, parent_id, file_id)
        self.create_file(contents, trans_id, sha1=sha1)
        if executable is not None:
            self.set_executability(executable, trans_id)
        return trans_id

    def new_directory(self, name, parent_id, file_id=None):
        """Convenience method to create directories.

        name is the name of the directory to create.
        parent_id is the transaction id of the parent directory of the
        directory.
        file_id is the inventory ID of the directory, if it is to be versioned.
        """
        trans_id = self._new_entry(name, parent_id, file_id)
        self.create_directory(trans_id)
        return trans_id

    def new_symlink(self, name, parent_id, target, file_id=None):
        """Convenience method to create symbolic link.

        name is the name of the symlink to create.
        parent_id is the transaction id of the parent directory of the symlink.
        target is a bytestring of the target of the symlink.
        file_id is the inventory ID of the file, if it is to be versioned.
        """
        trans_id = self._new_entry(name, parent_id, file_id)
        self.create_symlink(target, trans_id)
        return trans_id

    def new_orphan(self, trans_id, parent_id):
        """Schedule an item to be orphaned.

        When a directory is about to be removed, its children, if they are not
        versioned are moved out of the way: they don't have a parent anymore.

        :param trans_id: The trans_id of the existing item.
        :param parent_id: The parent trans_id of the item.
        """
        raise NotImplementedError(self.new_orphan)

    def _get_potential_orphans(self, dir_id):
        """Find the potential orphans in a directory.

        A directory can't be safely deleted if there are versioned files in it.
        If all the contained files are unversioned then they can be orphaned.

        The 'None' return value means that the directory contains at least one
        versioned file and should not be deleted.

        :param dir_id: The directory trans id.

        :return: A list of the orphan trans ids or None if at least one
             versioned file is present.
        """
        orphans = []
        for child_tid in self.by_parent()[dir_id]:
            if child_tid in self._removed_contents:
                continue
            if not self.final_is_versioned(child_tid):
                orphans.append(child_tid)
            else:
                orphans = None
                break
        return orphans

    def _affected_ids(self):
        """Return the set of transform ids affected by the transform"""
        trans_ids = set(self._removed_id)
        trans_ids.update(self._versioned)
        trans_ids.update(self._removed_contents)
        trans_ids.update(self._new_contents)
        trans_ids.update(self._new_executability)
        trans_ids.update(self._new_name)
        trans_ids.update(self._new_parent)
        return trans_ids

    def iter_changes(self, want_unversioned=False):
        """Produce output in the same format as Tree.iter_changes.

        Will produce nonsensical results if invoked while inventory/filesystem
        conflicts (as reported by TreeTransform.find_raw_conflicts()) are present.
        """
        final_paths = FinalPaths(self)
        trans_ids = self._affected_ids()
        results = []
        for trans_id in trans_ids:
            from_path = self.tree_path(trans_id)
            modified = False
            if from_path is None:
                from_versioned = False
            else:
                from_versioned = self._tree.is_versioned(from_path)
            if not want_unversioned and (not from_versioned):
                from_path = None
            to_path = final_paths.get_path(trans_id)
            if to_path is None:
                to_versioned = False
            else:
                to_versioned = self.final_is_versioned(trans_id)
            if not want_unversioned and (not to_versioned):
                to_path = None
            if from_versioned:
                from_entry = next(self._tree.iter_entries_by_dir(specific_files=[from_path]))[1]
                from_name = from_entry.name
            else:
                from_entry = None
                if from_path is None:
                    from_name = None
                else:
                    from_name = os.path.basename(from_path)
            if from_path is not None:
                from_kind, from_executable, from_stats = self._tree._comparison_data(from_entry, from_path)
            else:
                from_kind = None
                from_executable = False
            to_name = self.final_name(trans_id)
            to_kind = self.final_kind(trans_id)
            if trans_id in self._new_executability:
                to_executable = self._new_executability[trans_id]
            else:
                to_executable = from_executable
            if from_versioned and from_kind != to_kind:
                modified = True
            elif to_kind in ('file', 'symlink') and trans_id in self._new_contents:
                modified = True
            if not modified and from_versioned == to_versioned and (from_path == to_path) and (from_name == to_name) and (from_executable == to_executable):
                continue
            if (from_path, to_path) == (None, None):
                continue
            results.append(TreeChange((from_path, to_path), modified, (from_versioned, to_versioned), (from_name, to_name), (from_kind, to_kind), (from_executable, to_executable)))

        def path_key(c):
            return (c.path[0] or '', c.path[1] or '')
        return iter(sorted(results, key=path_key))

    def get_preview_tree(self):
        """Return a tree representing the result of the transform.

        The tree is a snapshot, and altering the TreeTransform will invalidate
        it.
        """
        return GitPreviewTree(self)

    def commit(self, branch, message, merge_parents=None, strict=False, timestamp=None, timezone=None, committer=None, authors=None, revprops=None, revision_id=None):
        """Commit the result of this TreeTransform to a branch.

        :param branch: The branch to commit to.
        :param message: The message to attach to the commit.
        :param merge_parents: Additional parent revision-ids specified by
            pending merges.
        :param strict: If True, abort the commit if there are unversioned
            files.
        :param timestamp: if not None, seconds-since-epoch for the time and
            date.  (May be a float.)
        :param timezone: Optional timezone for timestamp, as an offset in
            seconds.
        :param committer: Optional committer in email-id format.
            (e.g. "J Random Hacker <jrandom@example.com>")
        :param authors: Optional list of authors in email-id format.
        :param revprops: Optional dictionary of revision properties.
        :param revision_id: Optional revision id.  (Specifying a revision-id
            may reduce performance for some non-native formats.)
        :return: The revision_id of the revision committed.
        """
        self._check_malformed()
        if strict:
            unversioned = set(self._new_contents).difference(set(self._versioned))
            for trans_id in unversioned:
                if not self.final_is_versioned(trans_id):
                    raise errors.StrictCommitFailed()
        revno, last_rev_id = branch.last_revision_info()
        if last_rev_id == _mod_revision.NULL_REVISION:
            if merge_parents is not None:
                raise ValueError('Cannot supply merge parents for first commit.')
            parent_ids = []
        else:
            parent_ids = [last_rev_id]
            if merge_parents is not None:
                parent_ids.extend(merge_parents)
        if self._tree.get_revision_id() != last_rev_id:
            raise ValueError('TreeTransform not based on branch basis: %s' % self._tree.get_revision_id().decode('utf-8'))
        from .. import commit
        revprops = commit.Commit.update_revprops(revprops, branch, authors)
        builder = branch.get_commit_builder(parent_ids, timestamp=timestamp, timezone=timezone, committer=committer, revprops=revprops, revision_id=revision_id)
        preview = self.get_preview_tree()
        list(builder.record_iter_changes(preview, last_rev_id, self.iter_changes()))
        builder.finish_inventory()
        revision_id = builder.commit(message)
        branch.set_last_revision_info(revno + 1, revision_id)
        return revision_id

    def _text_parent(self, trans_id):
        path = self.tree_path(trans_id)
        try:
            if path is None or self._tree.kind(path) != 'file':
                return None
        except _mod_transport.NoSuchFile:
            return None
        return path

    def _get_parents_texts(self, trans_id):
        """Get texts for compression parents of this file."""
        path = self._text_parent(trans_id)
        if path is None:
            return ()
        return (self._tree.get_file_text(path),)

    def _get_parents_lines(self, trans_id):
        """Get lines for compression parents of this file."""
        path = self._text_parent(trans_id)
        if path is None:
            return ()
        return (self._tree.get_file_lines(path),)

    def create_file(self, contents, trans_id, mode_id=None, sha1=None):
        """Schedule creation of a new file.

        :seealso: new_file.

        :param contents: an iterator of strings, all of which will be written
            to the target destination.
        :param trans_id: TreeTransform handle
        :param mode_id: If not None, force the mode of the target file to match
            the mode of the object referenced by mode_id.
            Otherwise, we will try to preserve mode bits of an existing file.
        :param sha1: If the sha1 of this content is already known, pass it in.
            We can use it to prevent future sha1 computations.
        """
        raise NotImplementedError(self.create_file)

    def create_directory(self, trans_id):
        """Schedule creation of a new directory.

        See also new_directory.
        """
        raise NotImplementedError(self.create_directory)

    def create_symlink(self, target, trans_id):
        """Schedule creation of a new symbolic link.

        target is a bytestring.
        See also new_symlink.
        """
        raise NotImplementedError(self.create_symlink)

    def create_tree_reference(self, target, trans_id):
        raise NotImplementedError(self.create_tree_reference)

    def create_hardlink(self, path, trans_id):
        """Schedule creation of a hard link"""
        raise NotImplementedError(self.create_hardlink)

    def cancel_creation(self, trans_id):
        """Cancel the creation of new file contents."""
        raise NotImplementedError(self.cancel_creation)

    def apply(self, no_conflicts=False, _mover=None):
        """Apply all changes to the inventory and filesystem.

        If filesystem or inventory conflicts are present, MalformedTransform
        will be thrown.

        If apply succeeds, finalize is not necessary.

        :param no_conflicts: if True, the caller guarantees there are no
            conflicts, so no check is made.
        :param _mover: Supply an alternate FileMover, for testing
        """
        raise NotImplementedError(self.apply)

    def cook_conflicts(self, raw_conflicts):
        """Generate a list of cooked conflicts, sorted by file path"""
        if not raw_conflicts:
            return
        fp = FinalPaths(self)
        from .workingtree import TextConflict, ContentsConflict
        for c in raw_conflicts:
            if c[0] == 'text conflict':
                yield TextConflict(fp.get_path(c[1]))
            elif c[0] == 'contents conflict':
                yield ContentsConflict(fp.get_path(c[1][0]))
            elif c[0] == 'duplicate':
                yield TextConflict(fp.get_path(c[2]))
            elif c[0] == 'missing parent':
                pass
            elif c[0] == 'non-directory parent':
                yield TextConflict(fp.get_path(c[2]))
            elif c[0] == 'deleting parent':
                yield TextConflict(fp.get_path(c[2]))
            elif c[0] == 'parent loop':
                yield TextConflict(fp.get_path(c[2]))
            else:
                raise AssertionError('unknown conflict %s' % c[0])