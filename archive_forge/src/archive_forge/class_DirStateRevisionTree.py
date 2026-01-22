import os
from io import BytesIO
from ..lazy_import import lazy_import
import contextlib
import errno
import stat
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import revision as _mod_revision
from ..lock import LogicalLockResult
from ..lockable_files import LockableFiles
from ..lockdir import LockDir
from ..mutabletree import BadReferenceTarget, MutableTree
from ..osutils import file_kind, isdir, pathjoin, realpath, safe_unicode
from ..transport import NoSuchFile, get_transport_from_path
from ..transport.local import LocalTransport
from ..tree import FileTimestampUnavailable, InterTree, MissingNestedTree
from ..workingtree import WorkingTree
from . import dirstate
from .inventory import ROOT_ID, Inventory, entry_factory
from .inventorytree import (InterInventoryTree, InventoryRevisionTree,
from .workingtree import InventoryWorkingTree, WorkingTreeFormatMetaDir
class DirStateRevisionTree(InventoryTree):
    """A revision tree pulling the inventory from a dirstate.

    Note that this is one of the historical (ie revision) trees cached in the
    dirstate for easy access, not the workingtree.
    """

    def __init__(self, dirstate, revision_id, repository, nested_tree_transport):
        self._dirstate = dirstate
        self._revision_id = revision_id
        self._repository = repository
        self._inventory = None
        self._locked = 0
        self._dirstate_locked = False
        self._nested_tree_transport = nested_tree_transport
        self._repo_supports_tree_reference = getattr(repository._format, 'supports_tree_reference', False)

    def __repr__(self):
        return '<%s of %s in %s>' % (self.__class__.__name__, self._revision_id, self._dirstate)

    def annotate_iter(self, path, default_revision=_mod_revision.CURRENT_REVISION):
        """See Tree.annotate_iter"""
        file_id = self.path2id(path)
        text_key = (file_id, self.get_file_revision(path))
        annotations = self._repository.texts.annotate(text_key)
        return [(key[-1], line) for key, line in annotations]

    def _comparison_data(self, entry, path):
        """See Tree._comparison_data."""
        if entry is None:
            return (None, False, None)
        return (entry.kind, entry.executable, None)

    def _get_file_revision(self, path, file_id, vf, tree_revision):
        """Ensure that file_id, tree_revision is in vf to plan the merge."""
        last_revision = self.get_file_revision(path)
        base_vf = self._repository.texts
        if base_vf not in vf.fallback_versionedfiles:
            vf.fallback_versionedfiles.append(base_vf)
        return last_revision

    def filter_unversioned_files(self, paths):
        """Filter out paths that are not versioned.

        :return: set of paths.
        """
        pred = self.has_filename
        return {p for p in paths if not pred(p)}

    def id2path(self, file_id, recurse='down'):
        """Convert a file-id to a path."""
        with self.lock_read():
            entry = self._get_entry(file_id=file_id)
            if entry == (None, None):
                if recurse == 'down':
                    if 'evil' in debug.debug_flags:
                        trace.mutter_callsite(2, 'Tree.id2path scans all nested trees.')
                    for nested_path in self.iter_references():
                        nested_tree = self.get_nested_tree(nested_path)
                        try:
                            return osutils.pathjoin(nested_path, nested_tree.id2path(file_id))
                        except errors.NoSuchId:
                            pass
                raise errors.NoSuchId(tree=self, file_id=file_id)
            path_utf8 = osutils.pathjoin(entry[0][0], entry[0][1])
            return path_utf8.decode('utf8')

    def get_nested_tree(self, path):
        with self.lock_read():
            nested_revid = self.get_reference_revision(path)
            return self._get_nested_tree(path, None, nested_revid)

    def _get_nested_tree(self, path, file_id, reference_revision):
        try:
            branch = _mod_branch.Branch.open_from_transport(self._nested_tree_transport.clone(path))
        except errors.NotBranchError as e:
            raise MissingNestedTree(path) from e
        try:
            revtree = branch.repository.revision_tree(reference_revision)
        except errors.NoSuchRevision as e:
            raise MissingNestedTree(path) from e
        if file_id is not None and revtree.path2id('') != file_id:
            raise AssertionError('mismatching file id: {!r} != {!r}'.format(revtree.path2id(''), file_id))
        return revtree

    def iter_references(self):
        if not self._repo_supports_tree_reference:
            return iter([])
        return super().iter_references()

    def _get_parent_index(self):
        """Return the index in the dirstate referenced by this tree."""
        return self._dirstate.get_parent_ids().index(self._revision_id) + 1

    def _get_entry(self, file_id=None, path=None):
        """Get the dirstate row for file_id or path.

        If either file_id or path is supplied, it is used as the key to lookup.
        If both are supplied, the fastest lookup is used, and an error is
        raised if they do not both point at the same row.

        :param file_id: An optional unicode file_id to be looked up.
        :param path: An optional unicode path to be looked up.
        :return: The dirstate row tuple for path/file_id, or (None, None)
        """
        if file_id is None and path is None:
            raise errors.BzrError('must supply file_id or path')
        if path is not None:
            path = path.encode('utf8')
        try:
            parent_index = self._get_parent_index()
        except ValueError:
            raise errors.NoSuchRevisionInTree(self._dirstate, self._revision_id)
        return self._dirstate._get_entry(parent_index, fileid_utf8=file_id, path_utf8=path)

    def _generate_inventory(self):
        """Create and set self.inventory from the dirstate object.

        (So this is only called the first time the inventory is requested for
        this tree; it then remains in memory until it's out of date.)

        This is relatively expensive: we have to walk the entire dirstate.
        """
        if not self._locked:
            raise AssertionError('cannot generate inventory of an unlocked dirstate revision tree')
        self._dirstate._read_dirblocks_if_needed()
        if self._revision_id not in self._dirstate.get_parent_ids():
            raise AssertionError('parent {} has disappeared from {}'.format(self._revision_id, self._dirstate.get_parent_ids()))
        parent_index = self._dirstate.get_parent_ids().index(self._revision_id) + 1
        root_key, current_entry = self._dirstate._get_entry(parent_index, path_utf8=b'')
        current_id = root_key[2]
        if current_entry[parent_index][0] != b'd':
            raise AssertionError()
        inv = Inventory(root_id=current_id, revision_id=self._revision_id)
        inv.root.revision = current_entry[parent_index][4]
        minikind_to_kind = dirstate.DirState._minikind_to_kind
        factory = entry_factory
        utf8_decode = cache_utf8._utf8_decode
        inv_byid = inv._byid
        parent_ies = {b'': inv.root}
        for block in self._dirstate._dirblocks[1:]:
            dirname = block[0]
            try:
                parent_ie = parent_ies[dirname]
            except KeyError:
                continue
            for key, entry in block[1]:
                minikind, fingerprint, size, executable, revid = entry[parent_index]
                if minikind in (b'a', b'r'):
                    continue
                name = key[1]
                name_unicode = utf8_decode(name)[0]
                file_id = key[2]
                kind = minikind_to_kind[minikind]
                inv_entry = factory[kind](file_id, name_unicode, parent_ie.file_id)
                inv_entry.revision = revid
                if kind == 'file':
                    inv_entry.executable = executable
                    inv_entry.text_size = size
                    inv_entry.text_sha1 = fingerprint
                elif kind == 'directory':
                    parent_ies[(dirname + b'/' + name).strip(b'/')] = inv_entry
                elif kind == 'symlink':
                    inv_entry.symlink_target = utf8_decode(fingerprint)[0]
                elif kind == 'tree-reference':
                    inv_entry.reference_revision = fingerprint or None
                else:
                    raise AssertionError('cannot convert entry %r into an InventoryEntry' % entry)
                if file_id in inv_byid:
                    raise AssertionError('file_id %s already in inventory as %s' % (file_id, inv_byid[file_id]))
                if name_unicode in parent_ie.children:
                    raise AssertionError('name %r already in parent' % (name_unicode,))
                inv_byid[file_id] = inv_entry
                parent_ie.children[name_unicode] = inv_entry
        self._inventory = inv

    def get_file_mtime(self, path):
        """Return the modification time for this record.

        We return the timestamp of the last-changed revision.
        """
        entry = self._get_entry(path=path)
        if entry == (None, None):
            nested_tree, subpath = self.get_containing_nested_tree(path)
            if nested_tree is not None:
                return nested_tree.get_file_mtime(subpath)
            raise NoSuchFile(path)
        parent_index = self._get_parent_index()
        last_changed_revision = entry[1][parent_index][4]
        try:
            rev = self._repository.get_revision(last_changed_revision)
        except errors.NoSuchRevision:
            raise FileTimestampUnavailable(path)
        return rev.timestamp

    def get_file_sha1(self, path, stat_value=None):
        entry = self._get_entry(path=path)
        parent_index = self._get_parent_index()
        parent_details = entry[1][parent_index]
        if parent_details[0] == b'f':
            return parent_details[1]
        return None

    def get_file_revision(self, path):
        with self.lock_read():
            inv, inv_file_id = self._path2inv_file_id(path)
            return inv.get_entry(inv_file_id).revision

    def get_file(self, path):
        return BytesIO(self.get_file_text(path))

    def get_file_size(self, path):
        """See Tree.get_file_size"""
        inv, inv_file_id = self._path2inv_file_id(path)
        return inv.get_entry(inv_file_id).text_size

    def get_file_text(self, path):
        content = None
        for _, content_iter in self.iter_files_bytes([(path, None)]):
            if content is not None:
                raise AssertionError('iter_files_bytes returned too many entries')
            content = b''.join(content_iter)
        if content is None:
            raise AssertionError('iter_files_bytes did not return the requested data')
        return content

    def get_reference_revision(self, path):
        inv, inv_file_id = self._path2inv_file_id(path)
        return inv.get_entry(inv_file_id).reference_revision

    def iter_files_bytes(self, desired_files):
        """See Tree.iter_files_bytes.

        This version is implemented on top of Repository.iter_files_bytes"""
        parent_index = self._get_parent_index()
        repo_desired_files = []
        for path, identifier in desired_files:
            entry = self._get_entry(path=path)
            if entry == (None, None):
                raise NoSuchFile(path)
            repo_desired_files.append((entry[0][2], entry[1][parent_index][4], identifier))
        return self._repository.iter_files_bytes(repo_desired_files)

    def get_symlink_target(self, path):
        entry = self._get_entry(path=path)
        if entry is None:
            raise NoSuchFile(tree=self, path=path)
        parent_index = self._get_parent_index()
        if entry[1][parent_index][0] != b'l':
            return None
        else:
            target = entry[1][parent_index][1]
            target = target.decode('utf8')
            return target

    def get_revision_id(self):
        """Return the revision id for this tree."""
        return self._revision_id

    def _get_root_inventory(self):
        if self._inventory is not None:
            return self._inventory
        self._must_be_locked()
        self._generate_inventory()
        return self._inventory
    root_inventory = property(_get_root_inventory, doc='Inventory of this Tree')

    def get_parent_ids(self):
        """The parents of a tree in the dirstate are not cached."""
        return self._repository.get_revision(self._revision_id).parent_ids

    def has_filename(self, filename):
        return bool(self.path2id(filename))

    def kind(self, path):
        entry = self._get_entry(path=path)[1]
        if entry is None:
            raise NoSuchFile(path)
        parent_index = self._get_parent_index()
        return dirstate.DirState._minikind_to_kind[entry[parent_index][0]]

    def stored_kind(self, path):
        """See Tree.stored_kind"""
        return self.kind(path)

    def path_content_summary(self, path):
        """See Tree.path_content_summary."""
        inv, inv_file_id = self._path2inv_file_id(path)
        if inv_file_id is None:
            return ('missing', None, None, None)
        entry = inv.get_entry(inv_file_id)
        kind = entry.kind
        if kind == 'file':
            return (kind, entry.text_size, entry.executable, entry.text_sha1)
        elif kind == 'symlink':
            return (kind, None, None, entry.symlink_target)
        else:
            return (kind, None, None, None)

    def is_executable(self, path):
        inv, inv_file_id = self._path2inv_file_id(path)
        if inv_file_id is None:
            raise NoSuchFile(path)
        ie = inv.get_entry(inv_file_id)
        if ie.kind != 'file':
            return False
        return ie.executable

    def is_locked(self):
        return self._locked

    def list_files(self, include_root=False, from_dir=None, recursive=True, recurse_nested=False):
        if from_dir is None:
            from_dir_id = None
            inv = self.root_inventory
        else:
            inv, from_dir_id = self._path2inv_file_id(from_dir)
            if from_dir_id is None:
                return iter([])

        def iter_entries(inv):
            entries = inv.iter_entries(from_dir=from_dir_id, recursive=recursive)
            if inv.root is not None and (not include_root) and (from_dir is None):
                next(entries)
            for path, entry in entries:
                if entry.kind == 'tree-reference' and recurse_nested:
                    subtree = self._get_nested_tree(path, entry.file_id, entry.reference_revision)
                    for subpath, status, kind, entry in subtree.list_files(include_root=True, recursive=recursive, recurse_nested=recurse_nested):
                        if subpath:
                            full_subpath = osutils.pathjoin(path, subpath)
                        else:
                            full_subpath = path
                        yield (full_subpath, status, kind, entry)
                else:
                    yield (path, 'V', entry.kind, entry)
        return iter_entries(inv)

    def lock_read(self):
        """Lock the tree for a set of operations.

        :return: A breezy.lock.LogicalLockResult.
        """
        if not self._locked:
            self._repository.lock_read()
            if self._dirstate._lock_token is None:
                self._dirstate.lock_read()
                self._dirstate_locked = True
        self._locked += 1
        return LogicalLockResult(self.unlock)

    def _must_be_locked(self):
        if not self._locked:
            raise errors.ObjectNotLocked(self)

    def path2id(self, path):
        """Return the id for path in this tree."""
        if isinstance(path, list):
            if path == []:
                path = ['']
            path = osutils.pathjoin(*path)
        with self.lock_read():
            entry = self._get_entry(path=path)
            if entry == (None, None):
                nested_tree, subpath = self.get_containing_nested_tree(path)
                if nested_tree is not None:
                    return nested_tree.path2id(subpath)
                return None
            return entry[0][2]

    def unlock(self):
        """Unlock, freeing any cache memory used during the lock."""
        self._locked -= 1
        if not self._locked:
            self._inventory = None
            self._locked = 0
            if self._dirstate_locked:
                self._dirstate.unlock()
                self._dirstate_locked = False
            self._repository.unlock()

    def supports_tree_reference(self):
        with self.lock_read():
            return self._repo_supports_tree_reference

    def walkdirs(self, prefix=''):
        _directory = 'directory'
        inv = self._get_root_inventory()
        top_id = inv.path2id(prefix)
        if top_id is None:
            pending = []
        else:
            pending = [(prefix, top_id)]
        while pending:
            dirblock = []
            relpath, file_id = pending.pop()
            if relpath:
                relroot = relpath + '/'
            else:
                relroot = ''
            entry = inv.get_entry(file_id)
            subdirs = []
            for name, child in entry.sorted_children():
                toppath = relroot + name
                dirblock.append((toppath, name, child.kind, None, child.kind))
                if child.kind == _directory:
                    subdirs.append((toppath, child.file_id))
            yield (relpath, dirblock)
            pending.extend(reversed(subdirs))