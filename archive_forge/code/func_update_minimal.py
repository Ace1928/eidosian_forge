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
def update_minimal(self, key, minikind, executable=False, fingerprint=b'', packed_stat=None, size=0, path_utf8=None, fullscan=False):
    """Update an entry to the state in tree 0.

        This will either create a new entry at 'key' or update an existing one.
        It also makes sure that any other records which might mention this are
        updated as well.

        :param key: (dir, name, file_id) for the new entry
        :param minikind: The type for the entry (b'f' == 'file', b'd' ==
                'directory'), etc.
        :param executable: Should the executable bit be set?
        :param fingerprint: Simple fingerprint for new entry: canonical-form
            sha1 for files, referenced revision id for subtrees, etc.
        :param packed_stat: Packed stat value for new entry.
        :param size: Size information for new entry
        :param path_utf8: key[0] + '/' + key[1], just passed in to avoid doing
                extra computation.
        :param fullscan: If True then a complete scan of the dirstate is being
            done and checking for duplicate rows should not be done. This
            should only be set by set_state_from_inventory and similar methods.

        If packed_stat and fingerprint are not given, they're invalidated in
        the entry.
        """
    block = self._find_block(key)[1]
    if packed_stat is None:
        packed_stat = DirState.NULLSTAT
    entry_index, present = self._find_entry_index(key, block)
    new_details = (minikind, fingerprint, size, executable, packed_stat)
    id_index = self._get_id_index()
    if not present:
        if not fullscan:
            low_index, _ = self._find_entry_index(key[0:2] + (b'',), block)
            while low_index < len(block):
                entry = block[low_index]
                if entry[0][0:2] == key[0:2]:
                    if entry[1][0][0] not in (b'a', b'r'):
                        self._raise_invalid((b'%s/%s' % key[0:2]).decode('utf8'), key[2], 'Attempt to add item at path already occupied by id %r' % entry[0][2])
                    low_index += 1
                else:
                    break
        existing_keys = id_index.get(key[2], ())
        if not existing_keys:
            new_entry = (key, [new_details] + self._empty_parent_info())
        else:
            new_entry = (key, [new_details])
            for other_key in tuple(existing_keys):
                other_block_index, present = self._find_block_index_from_key(other_key)
                if not present:
                    raise AssertionError('could not find block for {}'.format(other_key))
                other_block = self._dirblocks[other_block_index][1]
                other_entry_index, present = self._find_entry_index(other_key, other_block)
                if not present:
                    raise AssertionError('update_minimal: could not find other entry for %s' % (other_key,))
                if path_utf8 is None:
                    raise AssertionError('no path')
                other_entry = other_block[other_entry_index]
                other_entry[1][0] = (b'r', path_utf8, 0, False, b'')
                if self._maybe_remove_row(other_block, other_entry_index, id_index):
                    entry_index, _ = self._find_entry_index(key, block)
            num_present_parents = self._num_present_parents()
            if num_present_parents:
                other_key = list(existing_keys)[0]
            for lookup_index in range(1, num_present_parents + 1):
                update_block_index, present = self._find_block_index_from_key(other_key)
                if not present:
                    raise AssertionError('could not find block for {}'.format(other_key))
                update_entry_index, present = self._find_entry_index(other_key, self._dirblocks[update_block_index][1])
                if not present:
                    raise AssertionError('update_minimal: could not find entry for {}'.format(other_key))
                update_details = self._dirblocks[update_block_index][1][update_entry_index][1][lookup_index]
                if update_details[0] in (b'a', b'r'):
                    new_entry[1].append(update_details)
                else:
                    pointer_path = osutils.pathjoin(*other_key[0:2])
                    new_entry[1].append((b'r', pointer_path, 0, False, b''))
        block.insert(entry_index, new_entry)
        self._add_to_id_index(id_index, key)
    else:
        block[entry_index][1][0] = new_details
        if path_utf8 is None:
            raise AssertionError('no path')
        existing_keys = id_index.get(key[2], ())
        if key not in existing_keys:
            raise AssertionError('We found the entry in the blocks, but the key is not in the id_index. key: %s, existing_keys: %s' % (key, existing_keys))
        for entry_key in existing_keys:
            if entry_key != key:
                block_index, present = self._find_block_index_from_key(entry_key)
                if not present:
                    raise AssertionError('not present: %r', entry_key)
                entry_index, present = self._find_entry_index(entry_key, self._dirblocks[block_index][1])
                if not present:
                    raise AssertionError('not present: %r', entry_key)
                self._dirblocks[block_index][1][entry_index][1][0] = (b'r', path_utf8, 0, False, b'')
    if new_details[0] == b'd':
        subdir_key = (osutils.pathjoin(*key[0:2]), b'', b'')
        block_index, present = self._find_block_index_from_key(subdir_key)
        if not present:
            self._dirblocks.insert(block_index, (subdir_key[0], []))
    self._mark_modified()