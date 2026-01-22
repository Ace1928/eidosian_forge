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
class DiskTreeTransform(TreeTransformBase):
    """Tree transform storing its contents on disk."""

    def __init__(self, tree, limbodir, pb=None, case_sensitive=True):
        """Constructor.
        :param tree: The tree that will be transformed, but not necessarily
            the output tree.
        :param limbodir: A directory where new files can be stored until
            they are installed in their proper places
        :param pb: ignored
        :param case_sensitive: If True, the target of the transform is
            case sensitive, not just case preserving.
        """
        TreeTransformBase.__init__(self, tree, pb, case_sensitive)
        self._limbodir = limbodir
        self._deletiondir = None
        self._limbo_files = {}
        self._possibly_stale_limbo_files = set()
        self._limbo_children = {}
        self._limbo_children_names = {}
        self._needs_rename = set()
        self._creation_mtime = None
        self._create_symlinks = osutils.supports_symlinks(self._limbodir)

    def finalize(self):
        """Release the working tree lock, if held, clean up limbo dir.

        This is required if apply has not been invoked, but can be invoked
        even after apply.
        """
        if self._tree is None:
            return
        try:
            limbo_paths = list(self._limbo_files.values())
            limbo_paths.extend(self._possibly_stale_limbo_files)
            limbo_paths.sort(reverse=True)
            for path in limbo_paths:
                try:
                    osutils.delete_any(path)
                except OSError as e:
                    if e.errno != errno.ENOENT:
                        raise
            try:
                osutils.delete_any(self._limbodir)
            except OSError:
                raise ImmortalLimbo(self._limbodir)
            try:
                if self._deletiondir is not None:
                    osutils.delete_any(self._deletiondir)
            except OSError:
                raise errors.ImmortalPendingDeletion(self._deletiondir)
        finally:
            TreeTransformBase.finalize(self)

    def _limbo_supports_executable(self):
        """Check if the limbo path supports the executable bit."""
        return osutils.supports_executable(self._limbodir)

    def _limbo_name(self, trans_id):
        """Generate the limbo name of a file"""
        limbo_name = self._limbo_files.get(trans_id)
        if limbo_name is None:
            limbo_name = self._generate_limbo_path(trans_id)
            self._limbo_files[trans_id] = limbo_name
        return limbo_name

    def _generate_limbo_path(self, trans_id):
        """Generate a limbo path using the trans_id as the relative path.

        This is suitable as a fallback, and when the transform should not be
        sensitive to the path encoding of the limbo directory.
        """
        self._needs_rename.add(trans_id)
        return osutils.pathjoin(self._limbodir, trans_id)

    def adjust_path(self, name, parent, trans_id):
        previous_parent = self._new_parent.get(trans_id)
        previous_name = self._new_name.get(trans_id)
        super().adjust_path(name, parent, trans_id)
        if trans_id in self._limbo_files and trans_id not in self._needs_rename:
            self._rename_in_limbo([trans_id])
            if previous_parent != parent:
                self._limbo_children[previous_parent].remove(trans_id)
            if previous_parent != parent or previous_name != name:
                del self._limbo_children_names[previous_parent][previous_name]

    def _rename_in_limbo(self, trans_ids):
        """Fix limbo names so that the right final path is produced.

        This means we outsmarted ourselves-- we tried to avoid renaming
        these files later by creating them with their final names in their
        final parents.  But now the previous name or parent is no longer
        suitable, so we have to rename them.

        Even for trans_ids that have no new contents, we must remove their
        entries from _limbo_files, because they are now stale.
        """
        for trans_id in trans_ids:
            old_path = self._limbo_files[trans_id]
            self._possibly_stale_limbo_files.add(old_path)
            del self._limbo_files[trans_id]
            if trans_id not in self._new_contents:
                continue
            new_path = self._limbo_name(trans_id)
            os.rename(old_path, new_path)
            self._possibly_stale_limbo_files.remove(old_path)
            for descendant in self._limbo_descendants(trans_id):
                desc_path = self._limbo_files[descendant]
                desc_path = new_path + desc_path[len(old_path):]
                self._limbo_files[descendant] = desc_path

    def _limbo_descendants(self, trans_id):
        """Return the set of trans_ids whose limbo paths descend from this."""
        descendants = set(self._limbo_children.get(trans_id, []))
        for descendant in list(descendants):
            descendants.update(self._limbo_descendants(descendant))
        return descendants

    def _set_mode(self, trans_id, mode_id, typefunc):
        raise NotImplementedError(self._set_mode)

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
        name = self._limbo_name(trans_id)
        with open(name, 'wb') as f:
            unique_add(self._new_contents, trans_id, 'file')
            f.writelines(contents)
        self._set_mtime(name)
        self._set_mode(trans_id, mode_id, S_ISREG)
        if sha1 is not None:
            self._observed_sha1s[trans_id] = (sha1, osutils.lstat(name))

    def _read_symlink_target(self, trans_id):
        return os.readlink(self._limbo_name(trans_id))

    def _set_mtime(self, path):
        """All files that are created get the same mtime.

        This time is set by the first object to be created.
        """
        if self._creation_mtime is None:
            self._creation_mtime = time.time()
        os.utime(path, (self._creation_mtime, self._creation_mtime))

    def create_hardlink(self, path, trans_id):
        """Schedule creation of a hard link"""
        name = self._limbo_name(trans_id)
        try:
            os.link(path, name)
        except OSError as e:
            if e.errno != errno.EPERM:
                raise
            raise errors.HardLinkNotSupported(path)
        try:
            unique_add(self._new_contents, trans_id, 'file')
        except BaseException:
            os.unlink(name)
            raise

    def create_directory(self, trans_id):
        """Schedule creation of a new directory.

        See also new_directory.
        """
        os.mkdir(self._limbo_name(trans_id))
        unique_add(self._new_contents, trans_id, 'directory')

    def create_symlink(self, target, trans_id):
        """Schedule creation of a new symbolic link.

        target is a bytestring.
        See also new_symlink.
        """
        if self._create_symlinks:
            os.symlink(target, self._limbo_name(trans_id))
        else:
            try:
                path = FinalPaths(self).get_path(trans_id)
            except KeyError:
                path = None
            trace.warning('Unable to create symlink "{}" on this filesystem.'.format(path))
            self._symlink_target[trans_id] = target
        unique_add(self._new_contents, trans_id, 'symlink')

    def create_tree_reference(self, reference_revision, trans_id):
        """Schedule creation of a new symbolic link.

        target is a bytestring.
        See also new_symlink.
        """
        os.mkdir(self._limbo_name(trans_id))
        unique_add(self._new_reference_revision, trans_id, reference_revision)
        unique_add(self._new_contents, trans_id, 'tree-reference')

    def cancel_creation(self, trans_id):
        """Cancel the creation of new file contents."""
        del self._new_contents[trans_id]
        if trans_id in self._observed_sha1s:
            del self._observed_sha1s[trans_id]
        children = self._limbo_children.get(trans_id)
        if children is not None:
            self._rename_in_limbo(children)
            del self._limbo_children[trans_id]
            del self._limbo_children_names[trans_id]
        osutils.delete_any(self._limbo_name(trans_id))

    def new_orphan(self, trans_id, parent_id):
        conf = self._tree.get_config_stack()
        handle_orphan = conf.get('transform.orphan_policy')
        handle_orphan(self, trans_id, parent_id)

    def final_entry(self, trans_id):
        is_versioned = self.final_is_versioned(trans_id)
        fp = FinalPaths(self)
        tree_path = fp.get_path(trans_id)
        if trans_id in self._new_contents:
            path = self._limbo_name(trans_id)
            st = os.lstat(path)
            kind = mode_kind(st.st_mode)
            name = self.final_name(trans_id)
            file_id = self._tree.mapping.generate_file_id(tree_path)
            parent_id = self._tree.mapping.generate_file_id(os.path.dirname(tree_path))
            if kind == 'directory':
                return (GitTreeDirectory(file_id, self.final_name(trans_id), parent_id=parent_id), is_versioned)
            executable = mode_is_executable(st.st_mode)
            mode = object_mode(kind, executable)
            blob = blob_from_path_and_stat(encode_git_path(path), st)
            if kind == 'symlink':
                return (GitTreeSymlink(file_id, name, parent_id, decode_git_path(blob.data)), is_versioned)
            elif kind == 'file':
                return (GitTreeFile(file_id, name, executable=executable, parent_id=parent_id, git_sha1=blob.id, text_size=len(blob.data)), is_versioned)
            else:
                raise AssertionError(kind)
        elif trans_id in self._removed_contents:
            return (None, None)
        else:
            orig_path = self.tree_path(trans_id)
            if orig_path is None:
                return (None, None)
            file_id = self._tree.mapping.generate_file_id(tree_path)
            if tree_path == '':
                parent_id = None
            else:
                parent_id = self._tree.mapping.generate_file_id(os.path.dirname(tree_path))
            try:
                ie = next(self._tree.iter_entries_by_dir(specific_files=[orig_path]))[1]
                ie.file_id = file_id
                ie.parent_id = parent_id
                return (ie, is_versioned)
            except StopIteration:
                try:
                    if self.tree_kind(trans_id) == 'directory':
                        return (GitTreeDirectory(file_id, self.final_name(trans_id), parent_id=parent_id), is_versioned)
                except OSError as e:
                    if e.errno != errno.ENOTDIR:
                        raise
                return (None, None)

    def final_git_entry(self, trans_id):
        if trans_id in self._new_contents:
            path = self._limbo_name(trans_id)
            st = os.lstat(path)
            kind = mode_kind(st.st_mode)
            if kind == 'directory':
                return (None, None)
            executable = mode_is_executable(st.st_mode)
            mode = object_mode(kind, executable)
            blob = blob_from_path_and_stat(encode_git_path(path), st)
        elif trans_id in self._removed_contents:
            return (None, None)
        else:
            orig_path = self.tree_path(trans_id)
            kind = self._tree.kind(orig_path)
            executable = self._tree.is_executable(orig_path)
            mode = object_mode(kind, executable)
            if kind == 'symlink':
                contents = self._tree.get_symlink_target(orig_path)
            elif kind == 'file':
                contents = self._tree.get_file_text(orig_path)
            elif kind == 'directory':
                return (None, None)
            else:
                raise AssertionError(kind)
            blob = Blob.from_string(contents)
        return (blob, mode)