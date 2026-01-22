import errno
import os
import posixpath
import stat
from collections import deque
from functools import partial
from io import BytesIO
from typing import Union, List, Tuple, Set
from dulwich.config import ConfigFile as GitConfigFile
from dulwich.config import parse_submodules
from dulwich.diff_tree import RenameDetector, tree_changes
from dulwich.errors import NotTreeError
from dulwich.index import (Index, IndexEntry, blob_from_path_and_stat,
from dulwich.object_store import OverlayObjectStore, iter_tree_contents, BaseObjectStore
from dulwich.objects import S_IFGITLINK, S_ISGITLINK, ZERO_SHA, Blob, Tree, ObjectID
from .. import controldir as _mod_controldir
from .. import delta, errors, mutabletree, osutils, revisiontree, trace
from .. import transport as _mod_transport
from .. import tree as _mod_tree
from .. import urlutils, workingtree
from ..bzr.inventorytree import InventoryTreeChange
from ..revision import CURRENT_REVISION, NULL_REVISION
from ..transport import get_transport
from ..tree import MissingNestedTree, TreeEntry
from .mapping import (decode_git_path, default_mapping, encode_git_path,
class MutableGitIndexTree(mutabletree.MutableTree, GitTree):
    store: BaseObjectStore

    def __init__(self):
        self._lock_mode = None
        self._lock_count = 0
        self._versioned_dirs = None
        self._index_dirty = False
        self._submodules = None

    def git_snapshot(self, want_unversioned=False):
        return snapshot_workingtree(self, want_unversioned=want_unversioned)

    def is_versioned(self, path):
        with self.lock_read():
            path = encode_git_path(path.rstrip('/'))
            index, subpath = self._lookup_index(path)
            return subpath in index or self._has_dir(path)

    def _has_dir(self, path):
        if not isinstance(path, bytes):
            raise TypeError(path)
        if path == b'':
            return True
        if self._versioned_dirs is None:
            self._load_dirs()
        return path in self._versioned_dirs

    def _load_dirs(self):
        if self._lock_mode is None:
            raise errors.ObjectNotLocked(self)
        self._versioned_dirs = set()
        for p, sha, mode in self.iter_git_objects():
            self._ensure_versioned_dir(posixpath.dirname(p))

    def _ensure_versioned_dir(self, dirname):
        if not isinstance(dirname, bytes):
            raise TypeError(dirname)
        if dirname in self._versioned_dirs:
            return
        if dirname != b'':
            self._ensure_versioned_dir(posixpath.dirname(dirname))
        self._versioned_dirs.add(dirname)

    def path2id(self, path):
        with self.lock_read():
            path = path.rstrip('/')
            if self.is_versioned(path.rstrip('/')):
                return self.mapping.generate_file_id(osutils.safe_unicode(path))
            return None

    def _set_root_id(self, file_id):
        raise errors.UnsupportedOperation(self._set_root_id, self)

    def add(self, files, kinds=None):
        """Add paths to the set of versioned paths.

        Note that the command line normally calls smart_add instead,
        which can automatically recurse.

        This adds the files to the tree, so that they will be
        recorded by the next commit.

        Args:
          files: List of paths to add, relative to the base of the tree.
          kinds: Optional parameter to specify the kinds to be used for
            each file.
        """
        if isinstance(files, str):
            if not (kinds is None or isinstance(kinds, str)):
                raise AssertionError()
            files = [files]
            if kinds is not None:
                kinds = [kinds]
        files = [path.strip('/') for path in files]
        if kinds is None:
            kinds = [None] * len(files)
        elif not len(kinds) == len(files):
            raise AssertionError()
        with self.lock_tree_write():
            for f in files:
                if self.is_control_filename(f):
                    raise errors.ForbiddenControlFileError(filename=f)
                fp = osutils.splitpath(f)
            self._gather_kinds(files, kinds)
            for path, kind in zip(files, kinds):
                path, can_access = osutils.normalized_filename(path)
                if not can_access:
                    raise errors.InvalidNormalization(path)
                self._index_add_entry(path, kind)

    def _gather_kinds(self, files, kinds):
        """Helper function for add - sets the entries of kinds."""
        raise NotImplementedError(self._gather_kinds)

    def _read_submodule_head(self, path):
        raise NotImplementedError(self._read_submodule_head)

    def _lookup_index(self, encoded_path):
        if not isinstance(encoded_path, bytes):
            raise TypeError(encoded_path)
        if encoded_path in self.index:
            return (self.index, encoded_path)
        index = self.index
        remaining_path = encoded_path
        while True:
            parts = remaining_path.split(b'/')
            for i in range(1, len(parts)):
                basepath = b'/'.join(parts[:i])
                try:
                    value = index[basepath]
                except KeyError:
                    continue
                else:
                    if S_ISGITLINK(value.mode):
                        index = self._get_submodule_index(basepath)
                        remaining_path = b'/'.join(parts[i:])
                        break
                    else:
                        return (index, remaining_path)
            else:
                return (index, remaining_path)
        return (index, remaining_path)

    def _index_del_entry(self, index, path):
        del index[path]
        self._index_dirty = True

    def _apply_index_changes(self, changes):
        for path, kind, executability, reference_revision, symlink_target in changes:
            if kind is None or kind == 'directory':
                index, subpath = self._lookup_index(encode_git_path(path))
                try:
                    self._index_del_entry(index, subpath)
                except KeyError:
                    pass
                else:
                    self._versioned_dirs = None
            else:
                self._index_add_entry(path, kind, reference_revision=reference_revision, symlink_target=symlink_target)
        self.flush()

    def _index_add_entry(self, path, kind, reference_revision=None, symlink_target=None):
        if kind == 'directory':
            return
        elif kind == 'file':
            blob = Blob()
            try:
                file, stat_val = self.get_file_with_stat(path)
            except (_mod_transport.NoSuchFile, OSError):
                file = BytesIO()
                stat_val = os.stat_result((stat.S_IFREG | 420, 0, 0, 0, 0, 0, 0, 0, 0, 0))
            with file:
                blob.set_raw_string(file.read())
            if blob.id not in self.store:
                self.store.add_object(blob)
            hexsha = blob.id
        elif kind == 'symlink':
            blob = Blob()
            try:
                stat_val = self._lstat(path)
            except OSError:
                stat_val = os.stat_result((stat.S_IFLNK, 0, 0, 0, 0, 0, 0, 0, 0, 0))
            if symlink_target is None:
                symlink_target = self.get_symlink_target(path)
            blob.set_raw_string(encode_git_path(symlink_target))
            if blob.id not in self.store:
                self.store.add_object(blob)
            hexsha = blob.id
        elif kind == 'tree-reference':
            if reference_revision is not None:
                hexsha = self.branch.lookup_bzr_revision_id(reference_revision)[0]
            else:
                hexsha = self._read_submodule_head(path)
                if hexsha is None:
                    raise errors.NoCommits(path)
            try:
                stat_val = self._lstat(path)
            except OSError:
                stat_val = os.stat_result((S_IFGITLINK, 0, 0, 0, 0, 0, 0, 0, 0, 0))
            stat_val = os.stat_result((S_IFGITLINK,) + stat_val[1:])
        else:
            raise AssertionError("unknown kind '%s'" % kind)
        ensure_normalized_path(path)
        encoded_path = encode_git_path(path)
        if b'\r' in encoded_path or b'\n' in encoded_path:
            trace.mutter('ignoring path with invalid newline in it: %r', path)
            return
        index, index_path = self._lookup_index(encoded_path)
        index[index_path] = index_entry_from_stat(stat_val, hexsha)
        self._index_dirty = True
        if self._versioned_dirs is not None:
            self._ensure_versioned_dir(index_path)

    def iter_git_objects(self):
        for p, entry in self._recurse_index_entries():
            yield (p, entry.sha, entry.mode)

    def _recurse_index_entries(self, index=None, basepath=b'', recurse_nested=False):
        with self.lock_read():
            if index is None:
                index = self.index
            for path, value in index.items():
                if isinstance(value, ConflictedIndexEntry):
                    if value.this is None:
                        continue
                    mode = value.this.mode
                else:
                    mode = value.mode
                if S_ISGITLINK(mode) and recurse_nested:
                    subindex = self._get_submodule_index(path)
                    yield from self._recurse_index_entries(index=subindex, basepath=path, recurse_nested=recurse_nested)
                else:
                    yield (posixpath.join(basepath, path), value)

    def iter_entries_by_dir(self, specific_files=None, recurse_nested=False):
        with self.lock_read():
            if specific_files is not None:
                specific_files = set(specific_files)
            else:
                specific_files = None
            root_ie = self._get_dir_ie('', None)
            ret = {}
            if specific_files is None or '' in specific_files:
                ret['', ''] = root_ie
            dir_ids = {'': root_ie.file_id}
            for path, value in self._recurse_index_entries(recurse_nested=recurse_nested):
                if self.mapping.is_special_file(path):
                    continue
                path = decode_git_path(path)
                if specific_files is not None and path not in specific_files:
                    continue
                parent, name = posixpath.split(path)
                try:
                    file_ie = self._get_file_ie(name, path, value, None)
                except _mod_transport.NoSuchFile:
                    continue
                if specific_files is None:
                    for dir_path, dir_ie in self._add_missing_parent_ids(parent, dir_ids):
                        ret[posixpath.dirname(dir_path), dir_path] = dir_ie
                file_ie.parent_id = self.path2id(parent)
                ret[posixpath.dirname(path), path] = file_ie
            if specific_files:
                for path in specific_files:
                    key = (posixpath.dirname(path), path)
                    if key not in ret and self.is_versioned(path):
                        ret[key] = self._get_dir_ie(path, self.path2id(key[0]))
            for (_, path), ie in sorted(ret.items()):
                yield (path, ie)

    def iter_references(self):
        if self.supports_tree_reference():
            for path, entry in self.iter_entries_by_dir():
                if entry.kind == 'tree-reference':
                    yield path

    def _get_dir_ie(self, path: str, parent_id) -> GitTreeDirectory:
        file_id = self.path2id(path)
        return GitTreeDirectory(file_id, posixpath.basename(path).strip('/'), parent_id)

    def _get_file_ie(self, name: str, path: str, value: Union[IndexEntry, ConflictedIndexEntry], parent_id) -> Union[GitTreeSymlink, GitTreeDirectory, GitTreeFile, GitTreeSubmodule]:
        if not isinstance(name, str):
            raise TypeError(name)
        if not isinstance(path, str):
            raise TypeError(path)
        if isinstance(value, IndexEntry):
            mode = value.mode
            sha = value.sha
            size = value.size
        elif isinstance(value, ConflictedIndexEntry):
            if value.this is None:
                raise _mod_transport.NoSuchFile(path)
            mode = value.this.mode
            sha = value.this.sha
            size = value.this.size
        else:
            raise TypeError(value)
        file_id = self.path2id(path)
        if not isinstance(file_id, bytes):
            raise TypeError(file_id)
        kind = mode_kind(mode)
        ie = entry_factory[kind](file_id, name, parent_id, git_sha1=sha)
        if kind == 'symlink':
            ie.symlink_target = self.get_symlink_target(path)
        elif kind == 'tree-reference':
            ie.reference_revision = self.get_reference_revision(path)
        elif kind == 'directory':
            pass
        else:
            ie.git_sha1 = sha
            ie.text_size = size
            ie.executable = bool(stat.S_ISREG(mode) and stat.S_IEXEC & mode)
        return ie

    def _add_missing_parent_ids(self, path: str, dir_ids) -> List[Tuple[str, GitTreeDirectory]]:
        if path in dir_ids:
            return []
        parent = posixpath.dirname(path).strip('/')
        ret = self._add_missing_parent_ids(parent, dir_ids)
        parent_id = dir_ids[parent]
        ie = self._get_dir_ie(path, parent_id)
        dir_ids[path] = ie.file_id
        ret.append((path, ie))
        return ret

    def _comparison_data(self, entry, path):
        if entry is None:
            return (None, False, None)
        return (entry.kind, entry.executable, None)

    def _unversion_path(self, path):
        if self._lock_mode is None:
            raise errors.ObjectNotLocked(self)
        encoded_path = encode_git_path(path)
        count = 0
        index, subpath = self._lookup_index(encoded_path)
        try:
            self._index_del_entry(index, encoded_path)
        except KeyError:
            for p in list(index):
                if p.startswith(subpath + b'/'):
                    count += 1
                    self._index_del_entry(index, p)
        else:
            count = 1
        self._versioned_dirs = None
        return count

    def unversion(self, paths):
        with self.lock_tree_write():
            for path in paths:
                if self._unversion_path(path) == 0:
                    raise _mod_transport.NoSuchFile(path)
            self._versioned_dirs = None
            self.flush()

    def flush(self):
        pass

    def update_basis_by_delta(self, revid, delta):
        for old_path, new_path, file_id, ie in delta:
            if old_path is not None:
                index, old_subpath = self._lookup_index(encode_git_path(old_path))
                if old_subpath in index:
                    self._index_del_entry(index, old_subpath)
                    self._versioned_dirs = None
            if new_path is not None and ie.kind != 'directory':
                self._index_add_entry(new_path, ie.kind)
        self.flush()
        self._set_merges_from_parent_ids([])

    def move(self, from_paths, to_dir=None, after=None):
        rename_tuples = []
        with self.lock_tree_write():
            to_abs = self.abspath(to_dir)
            if not os.path.isdir(to_abs):
                raise errors.BzrMoveFailedError('', to_dir, errors.NotADirectory(to_abs))
            for from_rel in from_paths:
                from_tail = os.path.split(from_rel)[-1]
                to_rel = os.path.join(to_dir, from_tail)
                self.rename_one(from_rel, to_rel, after=after)
                rename_tuples.append((from_rel, to_rel))
            self.flush()
            return rename_tuples

    def rename_one(self, from_rel, to_rel, after=None):
        from_path = encode_git_path(from_rel)
        to_rel, can_access = osutils.normalized_filename(to_rel)
        if not can_access:
            raise errors.InvalidNormalization(to_rel)
        to_path = encode_git_path(to_rel)
        with self.lock_tree_write():
            if not after:
                after = not self.has_filename(from_rel) and self.has_filename(to_rel) and (not self.is_versioned(to_rel))
            if after:
                if not self.has_filename(to_rel):
                    raise errors.BzrMoveFailedError(from_rel, to_rel, _mod_transport.NoSuchFile(to_rel))
                if self.basis_tree().is_versioned(to_rel):
                    raise errors.BzrMoveFailedError(from_rel, to_rel, errors.AlreadyVersionedError(to_rel))
                kind = self.kind(to_rel)
            else:
                try:
                    to_kind = self.kind(to_rel)
                except _mod_transport.NoSuchFile:
                    exc_type = errors.BzrRenameFailedError
                    to_kind = None
                else:
                    exc_type = errors.BzrMoveFailedError
                if self.is_versioned(to_rel):
                    raise exc_type(from_rel, to_rel, errors.AlreadyVersionedError(to_rel))
                if not self.has_filename(from_rel):
                    raise errors.BzrMoveFailedError(from_rel, to_rel, _mod_transport.NoSuchFile(from_rel))
                kind = self.kind(from_rel)
                if not self.is_versioned(from_rel) and kind != 'directory':
                    raise exc_type(from_rel, to_rel, errors.NotVersionedError(from_rel))
                if self.has_filename(to_rel):
                    raise errors.RenameFailedFilesExist(from_rel, to_rel, _mod_transport.FileExists(to_rel))
                kind = self.kind(from_rel)
            if not after and kind != 'directory':
                index, from_subpath = self._lookup_index(from_path)
                if from_subpath not in index:
                    raise errors.BzrMoveFailedError(from_rel, to_rel, errors.NotVersionedError(path=from_rel))
            if not after:
                try:
                    self._rename_one(from_rel, to_rel)
                except OSError as e:
                    if e.errno == errno.ENOENT:
                        raise errors.BzrMoveFailedError(from_rel, to_rel, _mod_transport.NoSuchFile(to_rel))
                    raise
            if kind != 'directory':
                index, from_index_path = self._lookup_index(from_path)
                try:
                    self._index_del_entry(index, from_path)
                except KeyError:
                    pass
                self._index_add_entry(to_rel, kind)
            else:
                todo = [(p, i) for p, i in self._recurse_index_entries() if p.startswith(from_path + b'/')]
                for child_path, child_value in todo:
                    child_to_index, child_to_index_path = self._lookup_index(posixpath.join(to_path, posixpath.relpath(child_path, from_path)))
                    child_to_index[child_to_index_path] = child_value
                    self._index_dirty = True
                    child_from_index, child_from_index_path = self._lookup_index(child_path)
                    self._index_del_entry(child_from_index, child_from_index_path)
            self._versioned_dirs = None
            self.flush()

    def path_content_summary(self, path):
        """See Tree.path_content_summary."""
        try:
            stat_result = self._lstat(path)
        except OSError as e:
            if getattr(e, 'errno', None) == errno.ENOENT:
                return ('missing', None, None, None)
            raise
        kind = mode_kind(stat_result.st_mode)
        if kind == 'file':
            size = stat_result.st_size
            executable = self._is_executable_from_path_and_stat(path, stat_result)
            return ('file', size, executable, self._sha_from_stat(path, stat_result))
        elif kind == 'directory':
            if self._directory_is_tree_reference(path):
                kind = 'tree-reference'
            return (kind, None, None, None)
        elif kind == 'symlink':
            target = osutils.readlink(self.abspath(path))
            return ('symlink', None, None, target)
        else:
            return (kind, None, None, None)

    def stored_kind(self, relpath):
        if relpath == '':
            return 'directory'
        index, index_path = self._lookup_index(encode_git_path(relpath))
        if index is None:
            return None
        try:
            mode = index[index_path].mode
        except KeyError:
            for p in index:
                if osutils.is_inside(decode_git_path(index_path), decode_git_path(p)):
                    return 'directory'
            return None
        else:
            return mode_kind(mode)

    def kind(self, relpath):
        kind = osutils.file_kind(self.abspath(relpath))
        if kind == 'directory':
            if self._directory_is_tree_reference(relpath):
                return 'tree-reference'
            return 'directory'
        else:
            return kind

    def _live_entry(self, relpath):
        raise NotImplementedError(self._live_entry)

    def transform(self, pb=None):
        from .transform import GitTreeTransform
        return GitTreeTransform(self, pb=pb)

    def has_changes(self, _from_tree=None):
        """Quickly check that the tree contains at least one commitable change.

        :param _from_tree: tree to compare against to find changes (default to
            the basis tree and is intended to be used by tests).

        :return: True if a change is found. False otherwise
        """
        with self.lock_read():
            if len(self.get_parent_ids()) > 1:
                return True
            if _from_tree is None:
                _from_tree = self.basis_tree()
            changes = self.iter_changes(_from_tree)
            if self.supports_symlinks():
                try:
                    change = next(changes)
                    if change.path[1] == '':
                        next(changes)
                    return True
                except StopIteration:
                    return False
            else:
                changes = filter(lambda c: c[6][0] != 'symlink' and c[4] != (None, None), changes)
                try:
                    next(iter(changes))
                except StopIteration:
                    return False
                return True