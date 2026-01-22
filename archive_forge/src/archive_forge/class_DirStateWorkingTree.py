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
class DirStateWorkingTree(InventoryWorkingTree):

    def __init__(self, basedir, branch, _control_files=None, _format=None, _controldir=None):
        """Construct a WorkingTree for basedir.

        If the branch is not supplied, it is opened automatically.
        If the branch is supplied, it must be the branch for this basedir.
        (branch.base is not cross checked, because for remote branches that
        would be meaningless).
        """
        self._format = _format
        self.controldir = _controldir
        basedir = safe_unicode(basedir)
        trace.mutter('opening working tree %r', basedir)
        self._branch = branch
        self.basedir = realpath(basedir)
        self._control_files = _control_files
        self._transport = self._control_files._transport
        self._dirty = None
        self._dirstate = None
        self._inventory = None
        self._setup_directory_is_tree_reference()
        self._detect_case_handling()
        self._rules_searcher = None
        self.views = self._make_views()
        self._iter_changes = dirstate._process_entry
        self._repo_supports_tree_reference = getattr(self._branch.repository._format, 'supports_tree_reference', False)

    def _add(self, files, kinds, ids):
        """See MutableTree._add."""
        with self.lock_tree_write():
            state = self.current_dirstate()
            for f, file_id, kind in zip(files, ids, kinds):
                f = f.strip('/')
                if self.path2id(f):
                    if f == b'' and self.path2id(f) == ROOT_ID:
                        state.set_path_id(b'', generate_ids.gen_file_id(f))
                    continue
                if file_id is None:
                    file_id = generate_ids.gen_file_id(f)
                state.add(f, file_id, kind, None, b'')
            self._make_dirty(reset_inventory=True)

    def _get_check_refs(self):
        """Return the references needed to perform a check of this tree."""
        return [('trees', self.last_revision())]

    def _make_dirty(self, reset_inventory):
        """Make the tree state dirty.

        :param reset_inventory: True if the cached inventory should be removed
            (presuming there is one).
        """
        self._dirty = True
        if reset_inventory and self._inventory is not None:
            self._inventory = None

    def add_reference(self, sub_tree):
        with self.lock_tree_write():
            try:
                sub_tree_path = self.relpath(sub_tree.basedir)
            except errors.PathNotChild:
                raise BadReferenceTarget(self, sub_tree, 'Target not inside tree.')
            sub_tree_id = sub_tree.path2id('')
            if sub_tree_id == self.path2id(''):
                raise BadReferenceTarget(self, sub_tree, 'Trees have the same root id.')
            try:
                self.id2path(sub_tree_id)
            except errors.NoSuchId:
                pass
            else:
                raise BadReferenceTarget(self, sub_tree, 'Root id already present in tree')
            self._add([sub_tree_path], ['tree-reference'], [sub_tree_id])

    def break_lock(self):
        """Break a lock if one is present from another instance.

        Uses the ui factory to ask for confirmation if the lock may be from
        an active process.

        This will probe the repository for its lock as well.
        """
        try:
            if self._dirstate is None:
                clear = True
            else:
                clear = False
            state = self._current_dirstate()
            if state._lock_token is not None:
                raise errors.LockActive(self.basedir)
            else:
                try:
                    state.lock_write()
                except errors.LockContention:
                    raise errors.LockActive(self.basedir)
                else:
                    state.unlock()
        finally:
            if clear:
                self._dirstate = None
        self._control_files.break_lock()
        self.branch.break_lock()

    def _comparison_data(self, entry, path):
        kind, executable, stat_value = WorkingTree._comparison_data(self, entry, path)
        if self._repo_supports_tree_reference and kind == 'directory' and (entry is not None) and (entry.kind == 'tree-reference'):
            kind = 'tree-reference'
        return (kind, executable, stat_value)

    def commit(self, message=None, revprops=None, *args, **kwargs):
        with self.lock_write():
            result = WorkingTree.commit(self, message, revprops, *args, **kwargs)
            self._make_dirty(reset_inventory=True)
            return result

    def current_dirstate(self):
        """Return the current dirstate object.

        This is not part of the tree interface and only exposed for ease of
        testing.

        :raises errors.NotWriteLocked: when not in a lock.
        """
        self._must_be_locked()
        return self._current_dirstate()

    def _current_dirstate(self):
        """Internal function that does not check lock status.

        This is needed for break_lock which also needs the dirstate.
        """
        if self._dirstate is not None:
            return self._dirstate
        local_path = self.controldir.get_workingtree_transport(None).local_abspath('dirstate')
        self._dirstate = dirstate.DirState.on_file(local_path, self._sha1_provider(), self._worth_saving_limit(), self._supports_executable())
        return self._dirstate

    def _sha1_provider(self):
        """A function that returns a SHA1Provider suitable for this tree.

        :return: None if content filtering is not supported by this tree.
          Otherwise, a SHA1Provider is returned that sha's the canonical
          form of files, i.e. after read filters are applied.
        """
        if self.supports_content_filtering():
            return ContentFilterAwareSHA1Provider(self)
        else:
            return None

    def _worth_saving_limit(self):
        """How many hash changes are ok before we must save the dirstate.

        :return: an integer. -1 means never save.
        """
        conf = self.get_config_stack()
        return conf.get('bzr.workingtree.worth_saving_limit')

    def filter_unversioned_files(self, paths):
        """Filter out paths that are versioned.

        :return: set of paths.
        """
        paths = sorted(paths)
        result = set()
        state = self.current_dirstate()
        for path in paths:
            dirname, basename = os.path.split(path.encode('utf8'))
            _, _, _, path_is_versioned = state._get_block_entry_index(dirname, basename, 0)
            if not path_is_versioned:
                result.add(path)
        return result

    def flush(self):
        """Write all cached data to disk."""
        if self._control_files._lock_mode != 'w':
            raise errors.NotWriteLocked(self)
        self.current_dirstate().save()
        self._inventory = None
        self._dirty = False

    def _gather_kinds(self, files, kinds):
        """See MutableTree._gather_kinds."""
        with self.lock_tree_write():
            for pos, f in enumerate(files):
                if kinds[pos] is None:
                    kinds[pos] = self.kind(f)

    def _generate_inventory(self):
        """Create and set self.inventory from the dirstate object.

        This is relatively expensive: we have to walk the entire dirstate.
        Ideally we would not, and can deprecate this function.
        """
        state = self.current_dirstate()
        state._read_dirblocks_if_needed()
        root_key, current_entry = self._get_entry(path='')
        current_id = root_key[2]
        if not current_entry[0][0] == b'd':
            raise AssertionError(current_entry)
        inv = Inventory(root_id=current_id)
        minikind_to_kind = dirstate.DirState._minikind_to_kind
        factory = entry_factory
        utf8_decode = cache_utf8._utf8_decode
        inv_byid = inv._byid
        parent_ies = {b'': inv.root}
        for block in state._dirblocks[1:]:
            dirname = block[0]
            try:
                parent_ie = parent_ies[dirname]
            except KeyError:
                continue
            for key, entry in block[1]:
                minikind, link_or_sha1, size, executable, stat = entry[0]
                if minikind in (b'a', b'r'):
                    continue
                name = key[1]
                name_unicode = utf8_decode(name)[0]
                file_id = key[2]
                kind = minikind_to_kind[minikind]
                inv_entry = factory[kind](file_id, name_unicode, parent_ie.file_id)
                if kind == 'file':
                    inv_entry.executable = executable
                elif kind == 'directory':
                    parent_ies[(dirname + b'/' + name).strip(b'/')] = inv_entry
                elif kind == 'tree-reference':
                    inv_entry.reference_revision = link_or_sha1 or None
                elif kind != 'symlink':
                    raise AssertionError('unknown kind %r' % kind)
                if file_id in inv_byid:
                    raise AssertionError('file_id %s already in inventory as %s' % (file_id, inv_byid[file_id]))
                if name_unicode in parent_ie.children:
                    raise AssertionError('name %r already in parent' % (name_unicode,))
                inv_byid[file_id] = inv_entry
                parent_ie.children[name_unicode] = inv_entry
        self._inventory = inv

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
        state = self.current_dirstate()
        if path is not None:
            path = path.encode('utf8')
        return state._get_entry(0, fileid_utf8=file_id, path_utf8=path)

    def get_file_sha1(self, path, stat_value=None):
        entry = self._get_entry(path=path)
        if entry[0] is None:
            raise NoSuchFile(self, path)
        if path is None:
            path = pathjoin(entry[0][0], entry[0][1]).decode('utf8')
        file_abspath = self.abspath(path)
        state = self.current_dirstate()
        if stat_value is None:
            try:
                stat_value = osutils.lstat(file_abspath)
            except OSError as e:
                if e.errno == errno.ENOENT:
                    return None
                else:
                    raise
        link_or_sha1 = dirstate.update_entry(state, entry, file_abspath, stat_value=stat_value)
        if entry[1][0][0] == b'f':
            if link_or_sha1 is None:
                file_obj, statvalue = self.get_file_with_stat(path)
                try:
                    sha1 = osutils.sha_file(file_obj)
                finally:
                    file_obj.close()
                self._observed_sha1(path, (sha1, statvalue))
                return sha1
            else:
                return link_or_sha1
        return None

    def _get_root_inventory(self):
        """Get the inventory for the tree. This is only valid within a lock."""
        if 'evil' in debug.debug_flags:
            trace.mutter_callsite(2, 'accessing .inventory forces a size of tree translation.')
        if self._inventory is not None:
            return self._inventory
        self._must_be_locked()
        self._generate_inventory()
        return self._inventory
    root_inventory = property(_get_root_inventory, doc='Root inventory of this tree')

    def get_parent_ids(self):
        """See Tree.get_parent_ids.

        This implementation requests the ids list from the dirstate file.
        """
        with self.lock_read():
            return self.current_dirstate().get_parent_ids()

    def get_reference_revision(self, path):
        try:
            return self.get_nested_tree(path).last_revision()
        except MissingNestedTree:
            entry = self._get_entry(path=path)
            if entry == (None, None):
                raise NoSuchFile(self, path)
            return entry[1][0][1]

    def get_nested_tree(self, path):
        try:
            return WorkingTree.open(self.abspath(path))
        except errors.NotBranchError as e:
            raise MissingNestedTree(path)

    def id2path(self, file_id, recurse='down'):
        """Convert a file-id to a path."""
        with self.lock_read():
            state = self.current_dirstate()
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

    def _is_executable_from_path_and_stat_from_basis(self, path, stat_result):
        entry = self._get_entry(path=path)
        if entry == (None, None):
            return False
        return entry[1][0][3]

    def is_executable(self, path):
        """Test if a file is executable or not.

        Note: The caller is expected to take a read-lock before calling this.
        """
        if not self._supports_executable():
            entry = self._get_entry(path=path)
            if entry == (None, None):
                return False
            return entry[1][0][3]
        else:
            self._must_be_locked()
            mode = osutils.lstat(self.abspath(path)).st_mode
            return bool(stat.S_ISREG(mode) and stat.S_IEXEC & mode)

    def all_file_ids(self):
        """See Tree.iter_all_file_ids"""
        self._must_be_locked()
        result = set()
        for key, tree_details in self.current_dirstate()._iter_entries():
            if tree_details[0][0] in (b'a', b'r'):
                continue
            result.add(key[2])
        return result

    def all_versioned_paths(self):
        self._must_be_locked()
        return {path for path, entry in self.root_inventory.iter_entries(recursive=True)}

    def __iter__(self):
        """Iterate through file_ids for this tree.

        file_ids are in a WorkingTree if they are in the working inventory
        and the working file exists.
        """
        with self.lock_read():
            result = []
            for key, tree_details in self.current_dirstate()._iter_entries():
                if tree_details[0][0] in (b'a', b'r'):
                    continue
                path = pathjoin(self.basedir, key[0].decode('utf8'), key[1].decode('utf8'))
                if osutils.lexists(path):
                    result.append(key[2])
            return iter(result)

    def iter_references(self):
        if not self._repo_supports_tree_reference:
            return
        with self.lock_read():
            for key, tree_details in self.current_dirstate()._iter_entries():
                if tree_details[0][0] in (b'a', b'r'):
                    continue
                if not key[1]:
                    continue
                relpath = pathjoin(key[0].decode('utf8'), key[1].decode('utf8'))
                try:
                    if self.kind(relpath) == 'tree-reference':
                        yield relpath
                except NoSuchFile:
                    continue

    def _observed_sha1(self, path, sha_and_stat):
        """See MutableTree._observed_sha1."""
        state = self.current_dirstate()
        entry = self._get_entry(path=path)
        state._observed_sha1(entry, *sha_and_stat)

    def kind(self, relpath):
        abspath = self.abspath(relpath)
        kind = file_kind(abspath)
        if self._repo_supports_tree_reference and kind == 'directory':
            with self.lock_read():
                entry = self._get_entry(path=relpath)
                if entry[1] is not None:
                    if entry[1][0][0] == b't':
                        kind = 'tree-reference'
        return kind

    def _last_revision(self):
        """See Mutable.last_revision."""
        with self.lock_read():
            parent_ids = self.current_dirstate().get_parent_ids()
            if parent_ids:
                return parent_ids[0]
            else:
                return _mod_revision.NULL_REVISION

    def lock_read(self):
        """See Branch.lock_read, and WorkingTree.unlock.

        :return: A breezy.lock.LogicalLockResult.
        """
        self.branch.lock_read()
        try:
            self._control_files.lock_read()
            try:
                state = self.current_dirstate()
                if not state._lock_token:
                    state.lock_read()
                self._repo_supports_tree_reference = getattr(self.branch.repository._format, 'supports_tree_reference', False)
            except BaseException:
                self._control_files.unlock()
                raise
        except BaseException:
            self.branch.unlock()
            raise
        return LogicalLockResult(self.unlock)

    def _lock_self_write(self):
        """This should be called after the branch is locked."""
        try:
            self._control_files.lock_write()
            try:
                state = self.current_dirstate()
                if not state._lock_token:
                    state.lock_write()
                self._repo_supports_tree_reference = getattr(self.branch.repository._format, 'supports_tree_reference', False)
            except BaseException:
                self._control_files.unlock()
                raise
        except BaseException:
            self.branch.unlock()
            raise
        return LogicalLockResult(self.unlock)

    def lock_tree_write(self):
        """See MutableTree.lock_tree_write, and WorkingTree.unlock.

        :return: A breezy.lock.LogicalLockResult.
        """
        self.branch.lock_read()
        return self._lock_self_write()

    def lock_write(self):
        """See MutableTree.lock_write, and WorkingTree.unlock.

        :return: A breezy.lock.LogicalLockResult.
        """
        self.branch.lock_write()
        return self._lock_self_write()

    def move(self, from_paths, to_dir, after=False):
        """See WorkingTree.move()."""
        result = []
        if not from_paths:
            return result
        with self.lock_tree_write():
            state = self.current_dirstate()
            if isinstance(from_paths, (str, bytes)):
                raise ValueError()
            to_dir_utf8 = to_dir.encode('utf8')
            to_entry_dirname, to_basename = os.path.split(to_dir_utf8)
            to_entry_block_index, to_entry_entry_index, dir_present, entry_present = state._get_block_entry_index(to_entry_dirname, to_basename, 0)
            if not entry_present:
                raise errors.BzrMoveFailedError('', to_dir, errors.NotVersionedError(to_dir))
            to_entry = state._dirblocks[to_entry_block_index][1][to_entry_entry_index]
            to_block_index = state._ensure_block(to_entry_block_index, to_entry_entry_index, to_dir_utf8)
            to_block = state._dirblocks[to_block_index]
            to_abs = self.abspath(to_dir)
            if not isdir(to_abs):
                raise errors.BzrMoveFailedError('', to_dir, errors.NotADirectory(to_abs))
            if to_entry[1][0][0] != b'd':
                raise errors.BzrMoveFailedError('', to_dir, errors.NotADirectory(to_abs))
            if self._inventory is not None:
                update_inventory = True
                inv = self.root_inventory
                to_dir_id = to_entry[0][2]
            else:
                update_inventory = False
            rollbacks = contextlib.ExitStack()

            def move_one(old_entry, from_path_utf8, minikind, executable, fingerprint, packed_stat, size, to_block, to_key, to_path_utf8):
                state._make_absent(old_entry)
                from_key = old_entry[0]
                rollbacks.callback(state.update_minimal, from_key, minikind, executable=executable, fingerprint=fingerprint, packed_stat=packed_stat, size=size, path_utf8=from_path_utf8)
                state.update_minimal(to_key, minikind, executable=executable, fingerprint=fingerprint, packed_stat=packed_stat, size=size, path_utf8=to_path_utf8)
                added_entry_index, _ = state._find_entry_index(to_key, to_block[1])
                new_entry = to_block[1][added_entry_index]
                rollbacks.callback(state._make_absent, new_entry)
            for from_rel in from_paths:
                from_rel_utf8 = from_rel.encode('utf8')
                from_dirname, from_tail = osutils.split(from_rel)
                from_dirname, from_tail_utf8 = osutils.split(from_rel_utf8)
                from_entry = self._get_entry(path=from_rel)
                if from_entry == (None, None):
                    raise errors.BzrMoveFailedError(from_rel, to_dir, errors.NotVersionedError(path=from_rel))
                from_id = from_entry[0][2]
                to_rel = pathjoin(to_dir, from_tail)
                to_rel_utf8 = pathjoin(to_dir_utf8, from_tail_utf8)
                item_to_entry = self._get_entry(path=to_rel)
                if item_to_entry != (None, None):
                    raise errors.BzrMoveFailedError(from_rel, to_rel, 'Target is already versioned.')
                if from_rel == to_rel:
                    raise errors.BzrMoveFailedError(from_rel, to_rel, 'Source and target are identical.')
                from_missing = not self.has_filename(from_rel)
                to_missing = not self.has_filename(to_rel)
                if after:
                    move_file = False
                else:
                    move_file = True
                if to_missing:
                    if not move_file:
                        raise errors.BzrMoveFailedError(from_rel, to_rel, NoSuchFile(path=to_rel, extra='New file has not been created yet'))
                    elif from_missing:
                        raise errors.BzrRenameFailedError(from_rel, to_rel, errors.PathsDoNotExist(paths=(from_rel, to_rel)))
                elif from_missing:
                    move_file = False
                elif not after:
                    raise errors.RenameFailedFilesExist(from_rel, to_rel)
                if move_file:
                    from_rel_abs = self.abspath(from_rel)
                    to_rel_abs = self.abspath(to_rel)
                    try:
                        osutils.rename(from_rel_abs, to_rel_abs)
                    except OSError as e:
                        raise errors.BzrMoveFailedError(from_rel, to_rel, e[1])
                    rollbacks.callback(osutils.rename, to_rel_abs, from_rel_abs)
                try:
                    if update_inventory:
                        from_entry = inv.get_entry(from_id)
                        current_parent = from_entry.parent_id
                        inv.rename(from_id, to_dir_id, from_tail)
                        rollbacks.callback(inv.rename, from_id, current_parent, from_tail)
                    old_block_index, old_entry_index, dir_present, file_present = state._get_block_entry_index(from_dirname, from_tail_utf8, 0)
                    old_block = state._dirblocks[old_block_index][1]
                    old_entry = old_block[old_entry_index]
                    from_key, old_entry_details = old_entry
                    cur_details = old_entry_details[0]
                    to_key = (to_block[0],) + from_key[1:3]
                    minikind = cur_details[0]
                    move_one(old_entry, from_path_utf8=from_rel_utf8, minikind=minikind, executable=cur_details[3], fingerprint=cur_details[1], packed_stat=cur_details[4], size=cur_details[2], to_block=to_block, to_key=to_key, to_path_utf8=to_rel_utf8)
                    if minikind == b'd':

                        def update_dirblock(from_dir, to_key, to_dir_utf8):
                            """Recursively update all entries in this dirblock."""
                            if from_dir == b'':
                                raise AssertionError('renaming root not supported')
                            from_key = (from_dir, '')
                            from_block_idx, present = state._find_block_index_from_key(from_key)
                            if not present:
                                return
                            from_block = state._dirblocks[from_block_idx]
                            to_block_index, to_entry_index, _, _ = state._get_block_entry_index(to_key[0], to_key[1], 0)
                            to_block_index = state._ensure_block(to_block_index, to_entry_index, to_dir_utf8)
                            to_block = state._dirblocks[to_block_index]
                            for entry in from_block[1][:]:
                                if not entry[0][0] == from_dir:
                                    raise AssertionError()
                                cur_details = entry[1][0]
                                to_key = (to_dir_utf8, entry[0][1], entry[0][2])
                                from_path_utf8 = osutils.pathjoin(entry[0][0], entry[0][1])
                                to_path_utf8 = osutils.pathjoin(to_dir_utf8, entry[0][1])
                                minikind = cur_details[0]
                                if minikind in (b'a', b'r'):
                                    continue
                                move_one(entry, from_path_utf8=from_path_utf8, minikind=minikind, executable=cur_details[3], fingerprint=cur_details[1], packed_stat=cur_details[4], size=cur_details[2], to_block=to_block, to_key=to_key, to_path_utf8=to_path_utf8)
                                if minikind == b'd':
                                    update_dirblock(from_path_utf8, to_key, to_path_utf8)
                        update_dirblock(from_rel_utf8, to_key, to_rel_utf8)
                except BaseException:
                    rollbacks.close()
                    raise
                result.append((from_rel, to_rel))
                state._mark_modified()
                self._make_dirty(reset_inventory=False)
            return result

    def _must_be_locked(self):
        if not self._control_files._lock_count:
            raise errors.ObjectNotLocked(self)

    def _new_tree(self):
        """Initialize the state in this tree to be a new tree."""
        self._dirty = True

    def path2id(self, path):
        """Return the id for path in this tree."""
        with self.lock_read():
            if isinstance(path, list):
                if path == []:
                    path = ['']
                path = osutils.pathjoin(*path)
            path = path.strip('/')
            entry = self._get_entry(path=path)
            if entry == (None, None):
                nested_tree, subpath = self.get_containing_nested_tree(path)
                if nested_tree is not None:
                    return nested_tree.path2id(subpath)
                return None
            return entry[0][2]

    def paths2ids(self, paths, trees=[], require_versioned=True):
        """See Tree.paths2ids().

        This specialisation fast-paths the case where all the trees are in the
        dirstate.
        """
        if paths is None:
            return None
        parents = self.get_parent_ids()
        for tree in trees:
            if not (isinstance(tree, DirStateRevisionTree) and tree._revision_id in parents):
                return super().paths2ids(paths, trees, require_versioned)
        search_indexes = [0] + [1 + parents.index(tree._revision_id) for tree in trees]
        paths_utf8 = set()
        for path in paths:
            paths_utf8.add(path.encode('utf8'))
        state = self.current_dirstate()
        if False and (state._dirblock_state == dirstate.DirState.NOT_IN_MEMORY and b'' not in paths):
            paths2ids = self._paths2ids_using_bisect
        else:
            paths2ids = self._paths2ids_in_memory
        return paths2ids(paths_utf8, search_indexes, require_versioned=require_versioned)

    def _paths2ids_in_memory(self, paths, search_indexes, require_versioned=True):
        state = self.current_dirstate()
        state._read_dirblocks_if_needed()

        def _entries_for_path(path):
            """Return a list with all the entries that match path for all ids.
            """
            dirname, basename = os.path.split(path)
            key = (dirname, basename, b'')
            block_index, present = state._find_block_index_from_key(key)
            if not present:
                return []
            result = []
            block = state._dirblocks[block_index][1]
            entry_index, _ = state._find_entry_index(key, block)
            while entry_index < len(block) and block[entry_index][0][0:2] == key[0:2]:
                result.append(block[entry_index])
                entry_index += 1
            return result
        if require_versioned:
            all_versioned = True
            for path in paths:
                path_entries = _entries_for_path(path)
                if not path_entries:
                    all_versioned = False
                    break
                found_versioned = False
                for entry in path_entries:
                    for index in search_indexes:
                        if entry[1][index][0] != b'a':
                            found_versioned = True
                            break
                if not found_versioned:
                    all_versioned = False
                    break
            if not all_versioned:
                raise errors.PathsNotVersionedError([p.decode('utf-8') for p in paths])
        search_paths = osutils.minimum_path_selection(paths)
        searched_paths = set()
        found_ids = set()

        def _process_entry(entry):
            """Look at search_indexes within entry.

            If a specific tree's details are relocated, add the relocation
            target to search_paths if not searched already. If it is absent, do
            nothing. Otherwise add the id to found_ids.
            """
            for index in search_indexes:
                if entry[1][index][0] == b'r':
                    if not osutils.is_inside_any(searched_paths, entry[1][index][1]):
                        search_paths.add(entry[1][index][1])
                elif entry[1][index][0] != b'a':
                    found_ids.add(entry[0][2])
        while search_paths:
            current_root = search_paths.pop()
            searched_paths.add(current_root)
            root_entries = _entries_for_path(current_root)
            if not root_entries:
                continue
            for entry in root_entries:
                _process_entry(entry)
            initial_key = (current_root, b'', b'')
            block_index, _ = state._find_block_index_from_key(initial_key)
            while block_index < len(state._dirblocks) and osutils.is_inside(current_root, state._dirblocks[block_index][0]):
                for entry in state._dirblocks[block_index][1]:
                    _process_entry(entry)
                block_index += 1
        return found_ids

    def _paths2ids_using_bisect(self, paths, search_indexes, require_versioned=True):
        state = self.current_dirstate()
        found_ids = set()
        split_paths = sorted((osutils.split(p) for p in paths))
        found = state._bisect_recursive(split_paths)
        if require_versioned:
            found_dir_names = {dir_name_id[:2] for dir_name_id in found}
            for dir_name in split_paths:
                if dir_name not in found_dir_names:
                    raise errors.PathsNotVersionedError([p.decode('utf-8') for p in paths])
        for dir_name_id, trees_info in found.items():
            for index in search_indexes:
                if trees_info[index][0] not in (b'r', b'a'):
                    found_ids.add(dir_name_id[2])
        return found_ids

    def read_working_inventory(self):
        """Read the working inventory.

        This is a meaningless operation for dirstate, but we obey it anyhow.
        """
        return self.root_inventory

    def revision_tree(self, revision_id):
        """See Tree.revision_tree.

        WorkingTree4 supplies revision_trees for any basis tree.
        """
        with self.lock_read():
            dirstate = self.current_dirstate()
            parent_ids = dirstate.get_parent_ids()
            if revision_id not in parent_ids:
                raise errors.NoSuchRevisionInTree(self, revision_id)
            if revision_id in dirstate.get_ghosts():
                raise errors.NoSuchRevisionInTree(self, revision_id)
            return DirStateRevisionTree(dirstate, revision_id, self.branch.repository, get_transport_from_path(self.basedir))

    def set_last_revision(self, new_revision):
        """Change the last revision in the working tree."""
        with self.lock_tree_write():
            parents = self.get_parent_ids()
            if new_revision in (_mod_revision.NULL_REVISION, None):
                if len(parents) >= 2:
                    raise AssertionError('setting the last parent to none with a pending merge is unsupported.')
                self.set_parent_ids([])
            else:
                self.set_parent_ids([new_revision] + parents[1:], allow_leftmost_as_ghost=True)

    def set_parent_ids(self, revision_ids, allow_leftmost_as_ghost=False):
        """Set the parent ids to revision_ids.

        See also set_parent_trees. This api will try to retrieve the tree data
        for each element of revision_ids from the trees repository. If you have
        tree data already available, it is more efficient to use
        set_parent_trees rather than set_parent_ids. set_parent_ids is however
        an easier API to use.

        :param revision_ids: The revision_ids to set as the parent ids of this
            working tree. Any of these may be ghosts.
        """
        with self.lock_tree_write():
            trees = []
            for revision_id in revision_ids:
                try:
                    revtree = self.branch.repository.revision_tree(revision_id)
                except (errors.NoSuchRevision, errors.RevisionNotPresent):
                    revtree = None
                trees.append((revision_id, revtree))
            self.set_parent_trees(trees, allow_leftmost_as_ghost=allow_leftmost_as_ghost)

    def set_parent_trees(self, parents_list, allow_leftmost_as_ghost=False):
        """Set the parents of the working tree.

        :param parents_list: A list of (revision_id, tree) tuples.
            If tree is None, then that element is treated as an unreachable
            parent tree - i.e. a ghost.
        """
        with self.lock_tree_write():
            dirstate = self.current_dirstate()
            if len(parents_list) > 0:
                if not allow_leftmost_as_ghost and parents_list[0][1] is None:
                    raise errors.GhostRevisionUnusableHere(parents_list[0][0])
            real_trees = []
            ghosts = []
            parent_ids = [rev_id for rev_id, tree in parents_list]
            graph = self.branch.repository.get_graph()
            heads = graph.heads(parent_ids)
            accepted_revisions = set()
            for rev_id, tree in parents_list:
                if len(accepted_revisions) > 0:
                    if rev_id in accepted_revisions or rev_id not in heads:
                        continue
                _mod_revision.check_not_reserved_id(rev_id)
                if tree is not None:
                    real_trees.append((rev_id, tree))
                else:
                    real_trees.append((rev_id, self.branch.repository.revision_tree(_mod_revision.NULL_REVISION)))
                    ghosts.append(rev_id)
                accepted_revisions.add(rev_id)
            updated = False
            if len(real_trees) == 1 and (not ghosts) and self.branch.repository._format.fast_deltas and isinstance(real_trees[0][1], InventoryRevisionTree) and self.get_parent_ids():
                rev_id, rev_tree = real_trees[0]
                basis_id = self.get_parent_ids()[0]
                try:
                    basis_tree = self.branch.repository.revision_tree(basis_id)
                except errors.NoSuchRevision:
                    pass
                else:
                    delta = rev_tree.root_inventory._make_delta(basis_tree.root_inventory)
                    dirstate.update_basis_by_delta(delta, rev_id)
                    updated = True
            if not updated:
                dirstate.set_parent_trees(real_trees, ghosts=ghosts)
            self._make_dirty(reset_inventory=False)

    def _set_root_id(self, file_id):
        """See WorkingTree.set_root_id."""
        state = self.current_dirstate()
        state.set_path_id(b'', file_id)
        if state._dirblock_state == dirstate.DirState.IN_MEMORY_MODIFIED:
            self._make_dirty(reset_inventory=True)

    def _sha_from_stat(self, path, stat_result):
        """Get a sha digest from the tree's stat cache.

        The default implementation assumes no stat cache is present.

        :param path: The path.
        :param stat_result: The stat result being looked up.
        """
        return self.current_dirstate().sha1_from_stat(path, stat_result)

    def supports_tree_reference(self):
        return self._repo_supports_tree_reference

    def unlock(self):
        """Unlock in format 4 trees needs to write the entire dirstate."""
        if self._control_files._lock_count == 1:
            self._cleanup()
            if self._control_files._lock_mode == 'w':
                if self._dirty:
                    self.flush()
            if self._dirstate is not None:
                self._dirstate.save()
                self._dirstate.unlock()
            self._dirstate = None
            self._inventory = None
        try:
            return self._control_files.unlock()
        finally:
            self.branch.unlock()

    def unversion(self, paths):
        """Remove the file ids in paths from the current versioned set.

        When a directory is unversioned, all of its children are automatically
        unversioned.

        :param paths: The file ids to stop versioning.
        :raises: NoSuchId if any fileid is not currently versioned.
        """
        with self.lock_tree_write():
            if not paths:
                return
            state = self.current_dirstate()
            state._read_dirblocks_if_needed()
            file_ids = set()
            for path in paths:
                file_id = self.path2id(path)
                if file_id is None:
                    raise NoSuchFile(self, path)
                file_ids.add(file_id)
            ids_to_unversion = set(file_ids)
            paths_to_unversion = set()
            for key, details in state._dirblocks[0][1]:
                if details[0][0] not in (b'a', b'r') and key[2] in ids_to_unversion:
                    raise errors.BzrError('Unversioning the / is not currently supported')
            block_index = 0
            while block_index < len(state._dirblocks):
                block = state._dirblocks[block_index]
                delete_block = False
                for path in paths_to_unversion:
                    if block[0].startswith(path) and (len(block[0]) == len(path) or block[0][len(path)] == '/'):
                        delete_block = True
                        break
                if delete_block:
                    entry_index = 0
                    while entry_index < len(block[1]):
                        entry = block[1][entry_index]
                        if entry[1][0][0] in (b'a', b'r'):
                            entry_index += 1
                        else:
                            ids_to_unversion.discard(entry[0][2])
                            if not state._make_absent(entry):
                                entry_index += 1
                    block_index += 1
                    continue
                entry_index = 0
                while entry_index < len(block[1]):
                    entry = block[1][entry_index]
                    if entry[1][0][0] in (b'a', b'r') or entry[0][2] not in ids_to_unversion:
                        entry_index += 1
                        continue
                    if entry[1][0][0] == b'd':
                        paths_to_unversion.add(pathjoin(entry[0][0], entry[0][1]))
                    if not state._make_absent(entry):
                        entry_index += 1
                    ids_to_unversion.remove(entry[0][2])
                block_index += 1
            if ids_to_unversion:
                raise errors.NoSuchId(self, next(iter(ids_to_unversion)))
            self._make_dirty(reset_inventory=False)
            if self._inventory is not None:
                for file_id in file_ids:
                    if self._inventory.has_id(file_id):
                        self._inventory.remove_recursive_id(file_id)

    def rename_one(self, from_rel, to_rel, after=False):
        """See WorkingTree.rename_one"""
        with self.lock_tree_write():
            self.flush()
            super().rename_one(from_rel, to_rel, after)

    def apply_inventory_delta(self, changes):
        """See MutableTree.apply_inventory_delta"""
        with self.lock_tree_write():
            state = self.current_dirstate()
            state.update_by_delta(changes)
            self._make_dirty(reset_inventory=True)

    def update_basis_by_delta(self, new_revid, delta):
        """See MutableTree.update_basis_by_delta."""
        if self.last_revision() == new_revid:
            raise AssertionError()
        self.current_dirstate().update_basis_by_delta(delta, new_revid)

    def _validate(self):
        with self.lock_read():
            self._dirstate._validate()

    def _write_inventory(self, inv):
        """Write inventory as the current inventory."""
        if self._dirty:
            raise AssertionError('attempting to write an inventory when the dirstate is dirty will lose pending changes')
        with self.lock_tree_write():
            had_inventory = self._inventory is not None
            self._inventory = None
            delta = inv._make_delta(self.root_inventory)
            self.apply_inventory_delta(delta)
            if had_inventory:
                self._inventory = inv
            self.flush()

    def reset_state(self, revision_ids=None):
        """Reset the state of the working tree.

        This does a hard-reset to a last-known-good state. This is a way to
        fix if something got corrupted (like the .bzr/checkout/dirstate file)
        """
        with self.lock_tree_write():
            if revision_ids is None:
                revision_ids = self.get_parent_ids()
            if not revision_ids:
                base_tree = self.branch.repository.revision_tree(_mod_revision.NULL_REVISION)
                trees = []
            else:
                trees = list(zip(revision_ids, self.branch.repository.revision_trees(revision_ids)))
                base_tree = trees[0][1]
            state = self.current_dirstate()
            state.set_state_from_scratch(base_tree.root_inventory, trees, [])