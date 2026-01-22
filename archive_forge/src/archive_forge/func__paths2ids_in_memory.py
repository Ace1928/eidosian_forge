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