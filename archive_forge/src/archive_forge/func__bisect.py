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
def _bisect(self, paths):
    """Bisect through the disk structure for specific rows.

        :param paths: A list of paths to find
        :return: A dict mapping path => entries for found entries. Missing
                 entries will not be in the map.
                 The list is not sorted, and entries will be populated
                 based on when they were read.
        """
    self._requires_lock()
    self._read_header_if_needed()
    if self._dirblock_state != DirState.NOT_IN_MEMORY:
        raise AssertionError('bad dirblock state %r' % self._dirblock_state)
    state_file = self._state_file
    file_size = os.fstat(state_file.fileno()).st_size
    entry_field_count = self._fields_per_entry() + 1
    low = self._end_of_header
    high = file_size - 1
    found = {}
    max_count = 30 * len(paths)
    count = 0
    pending = [(low, high, paths)]
    page_size = self._bisect_page_size
    fields_to_entry = self._get_fields_to_entry()
    while pending:
        low, high, cur_files = pending.pop()
        if not cur_files or low >= high:
            continue
        count += 1
        if count > max_count:
            raise errors.BzrError('Too many seeks, most likely a bug.')
        mid = max(low, (low + high - page_size) // 2)
        state_file.seek(mid)
        read_size = min(page_size, high - mid + 1)
        block = state_file.read(read_size)
        start = mid
        entries = block.split(b'\n')
        if len(entries) < 2:
            page_size *= 2
            pending.append((low, high, cur_files))
            continue
        first_entry_num = 0
        first_fields = entries[0].split(b'\x00')
        if len(first_fields) < entry_field_count:
            start += len(entries[0]) + 1
            first_fields = entries[1].split(b'\x00')
            first_entry_num = 1
        if len(first_fields) <= 2:
            page_size *= 2
            pending.append((low, high, cur_files))
            continue
        else:
            after = start
            if first_fields[1]:
                first_path = first_fields[1] + b'/' + first_fields[2]
            else:
                first_path = first_fields[2]
            first_loc = _bisect_path_left(cur_files, first_path)
            pre = cur_files[:first_loc]
            post = cur_files[first_loc:]
        if post and len(first_fields) >= entry_field_count:
            last_entry_num = len(entries) - 1
            last_fields = entries[last_entry_num].split(b'\x00')
            if len(last_fields) < entry_field_count:
                after = mid + len(block) - len(entries[-1])
                last_entry_num -= 1
                last_fields = entries[last_entry_num].split(b'\x00')
            else:
                after = mid + len(block)
            if last_fields[1]:
                last_path = last_fields[1] + b'/' + last_fields[2]
            else:
                last_path = last_fields[2]
            last_loc = _bisect_path_right(post, last_path)
            middle_files = post[:last_loc]
            post = post[last_loc:]
            if middle_files:
                if middle_files[0] == first_path:
                    pre.append(first_path)
                if middle_files[-1] == last_path:
                    post.insert(0, last_path)
                paths = {first_path: [first_fields]}
                if last_entry_num != first_entry_num:
                    paths.setdefault(last_path, []).append(last_fields)
                for num in range(first_entry_num + 1, last_entry_num):
                    fields = entries[num].split(b'\x00')
                    if fields[1]:
                        path = fields[1] + b'/' + fields[2]
                    else:
                        path = fields[2]
                    paths.setdefault(path, []).append(fields)
                for path in middle_files:
                    for fields in paths.get(path, []):
                        entry = fields_to_entry(fields[1:])
                        found.setdefault(path, []).append(entry)
        if post:
            pending.append((after, high, post))
        if pre:
            pending.append((low, start - 1, pre))
    return found