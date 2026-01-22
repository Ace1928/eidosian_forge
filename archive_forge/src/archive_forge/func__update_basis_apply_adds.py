import bisect
import codecs
import contextlib
import errno
import operator
import os
import stat
import sys
import time
import zlib
from stat import S_IEXEC
from .. import (cache_utf8, config, debug, errors, lock, osutils, trace,
from . import inventory, static_tuple
from .inventorytree import InventoryTreeChange
def _update_basis_apply_adds(self, adds):
    """Apply a sequence of adds to tree 1 during update_basis_by_delta.

        They may be adds, or renames that have been split into add/delete
        pairs.

        :param adds: A sequence of adds. Each add is a tuple:
            (None, new_path_utf8, file_id, (entry_details), real_add). real_add
            is False when the add is the second half of a remove-and-reinsert
            pair created to handle renames and deletes.
        """
    adds.sort(key=lambda x: x[1])
    st = static_tuple.StaticTuple
    for old_path, new_path, file_id, new_details, real_add in adds:
        dirname, basename = osutils.split(new_path)
        entry_key = st(dirname, basename, file_id)
        block_index, present = self._find_block_index_from_key(entry_key)
        if not present:
            parent_dir, parent_base = osutils.split(dirname)
            parent_block_idx, parent_entry_idx, _, parent_present = self._get_block_entry_index(parent_dir, parent_base, 1)
            if not parent_present:
                self._raise_invalid(new_path, file_id, 'Unable to find block for this record. Was the parent added?')
            self._ensure_block(parent_block_idx, parent_entry_idx, dirname)
        block = self._dirblocks[block_index][1]
        entry_index, present = self._find_entry_index(entry_key, block)
        if real_add:
            if old_path is not None:
                self._raise_invalid(new_path, file_id, 'considered a real add but still had old_path at %s' % (old_path,))
        if present:
            entry = block[entry_index]
            basis_kind = entry[1][1][0]
            if basis_kind == b'a':
                entry[1][1] = new_details
            elif basis_kind == b'r':
                raise NotImplementedError()
            else:
                self._raise_invalid(new_path, file_id, 'An entry was marked as a new add but the basis target already existed')
        else:
            for maybe_index in range(entry_index - 1, entry_index + 1):
                if maybe_index < 0 or maybe_index >= len(block):
                    continue
                maybe_entry = block[maybe_index]
                if maybe_entry[0][:2] != (dirname, basename):
                    continue
                if maybe_entry[0][2] == file_id:
                    raise AssertionError('_find_entry_index didnt find a key match but walking the data did, for %s' % (entry_key,))
                basis_kind = maybe_entry[1][1][0]
                if basis_kind not in (b'a', b'r'):
                    self._raise_invalid(new_path, file_id, 'we have an add record for path, but the path is already present with another file_id %s' % (maybe_entry[0][2],))
            entry = (entry_key, [DirState.NULL_PARENT_DETAILS, new_details])
            block.insert(entry_index, entry)
        active_kind = entry[1][0][0]
        if active_kind == b'a':
            id_index = self._get_id_index()
            keys = id_index.get(file_id, ())
            for key in keys:
                block_i, entry_i, d_present, f_present = self._get_block_entry_index(key[0], key[1], 0)
                if not f_present:
                    continue
                active_entry = self._dirblocks[block_i][1][entry_i]
                if active_entry[0][2] != file_id:
                    continue
                real_active_kind = active_entry[1][0][0]
                if real_active_kind in (b'a', b'r'):
                    self._raise_invalid(new_path, file_id, 'We found a tree0 entry that doesnt make sense')
                active_dir, active_name = active_entry[0][:2]
                if active_dir:
                    active_path = active_dir + b'/' + active_name
                else:
                    active_path = active_name
                active_entry[1][1] = st(b'r', new_path, 0, False, b'')
                entry[1][0] = st(b'r', active_path, 0, False, b'')
        elif active_kind == b'r':
            raise NotImplementedError()
        new_kind = new_details[0]
        if new_kind == b'd':
            self._ensure_block(block_index, entry_index, new_path)