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
def _process_entry(self, entry, path_info, pathjoin=osutils.pathjoin):
    """Compare an entry and real disk to generate delta information.

        :param path_info: top_relpath, basename, kind, lstat, abspath for
            the path of entry. If None, then the path is considered absent in
            the target (Perhaps we should pass in a concrete entry for this ?)
            Basename is returned as a utf8 string because we expect this
            tuple will be ignored, and don't want to take the time to
            decode.
        :return: (iter_changes_result, changed). If the entry has not been
            handled then changed is None. Otherwise it is False if no content
            or metadata changes have occurred, and True if any content or
            metadata change has occurred. If self.include_unchanged is True then
            if changed is not None, iter_changes_result will always be a result
            tuple. Otherwise, iter_changes_result is None unless changed is
            True.
        """
    if self.source_index is None:
        source_details = DirState.NULL_PARENT_DETAILS
    else:
        source_details = entry[1][self.source_index]
    _fdltr = {b'f', b'd', b'l', b't', b'r'}
    _fdlt = {b'f', b'd', b'l', b't'}
    _ra = (b'r', b'a')
    target_details = entry[1][self.target_index]
    target_minikind = target_details[0]
    if path_info is not None and target_minikind in _fdlt:
        if not self.target_index == 0:
            raise AssertionError()
        link_or_sha1 = update_entry(self.state, entry, abspath=path_info[4], stat_value=path_info[3])
        target_details = entry[1][self.target_index]
        target_minikind = target_details[0]
    else:
        link_or_sha1 = None
    file_id = entry[0][2]
    source_minikind = source_details[0]
    if source_minikind in _fdltr and target_minikind in _fdlt:
        if source_minikind == b'r':
            if not osutils.is_inside_any(self.searched_specific_files, source_details[1]):
                self.search_specific_files.add(source_details[1])
            old_path = source_details[1]
            old_dirname, old_basename = os.path.split(old_path)
            path = pathjoin(entry[0][0], entry[0][1])
            old_entry = self.state._get_entry(self.source_index, path_utf8=old_path)
            if old_entry == (None, None):
                raise DirstateCorrupt(self.state._filename, "entry '%s/%s' is considered renamed from %r but source does not exist\nentry: %s" % (entry[0][0], entry[0][1], old_path, entry))
            source_details = old_entry[1][self.source_index]
            source_minikind = source_details[0]
        else:
            old_dirname = entry[0][0]
            old_basename = entry[0][1]
            old_path = path = None
        if path_info is None:
            content_change = True
            target_kind = None
            target_exec = False
        else:
            target_kind = path_info[2]
            if target_kind == 'directory':
                if path is None:
                    old_path = path = pathjoin(old_dirname, old_basename)
                self.new_dirname_to_file_id[path] = file_id
                if source_minikind != b'd':
                    content_change = True
                else:
                    content_change = False
                target_exec = False
            elif target_kind == 'file':
                if source_minikind != b'f':
                    content_change = True
                else:
                    if link_or_sha1 is None:
                        statvalue, link_or_sha1 = self.state._sha1_provider.stat_and_sha1(path_info[4])
                        self.state._observed_sha1(entry, link_or_sha1, statvalue)
                    content_change = link_or_sha1 != source_details[1]
                if self.use_filesystem_for_exec:
                    target_exec = bool(stat.S_IEXEC & path_info[3].st_mode)
                else:
                    target_exec = target_details[3]
            elif target_kind == 'symlink':
                if source_minikind != b'l':
                    content_change = True
                else:
                    content_change = link_or_sha1 != source_details[1]
                target_exec = False
            elif target_kind == 'tree-reference':
                if source_minikind != b't':
                    content_change = True
                else:
                    content_change = False
                target_exec = False
            else:
                if path is None:
                    path = pathjoin(old_dirname, old_basename)
                raise errors.BadFileKindError(path, path_info[2])
        if source_minikind == b'd':
            if path is None:
                old_path = path = pathjoin(old_dirname, old_basename)
            self.old_dirname_to_file_id[old_path] = file_id
        if old_basename and old_dirname == self.last_source_parent[0]:
            source_parent_id = self.last_source_parent[1]
        else:
            try:
                source_parent_id = self.old_dirname_to_file_id[old_dirname]
            except KeyError:
                source_parent_entry = self.state._get_entry(self.source_index, path_utf8=old_dirname)
                source_parent_id = source_parent_entry[0][2]
            if source_parent_id == entry[0][2]:
                source_parent_id = None
            else:
                self.last_source_parent[0] = old_dirname
                self.last_source_parent[1] = source_parent_id
        new_dirname = entry[0][0]
        if entry[0][1] and new_dirname == self.last_target_parent[0]:
            target_parent_id = self.last_target_parent[1]
        else:
            try:
                target_parent_id = self.new_dirname_to_file_id[new_dirname]
            except KeyError:
                target_parent_entry = self.state._get_entry(self.target_index, path_utf8=new_dirname)
                if target_parent_entry == (None, None):
                    raise AssertionError('Could not find target parent in wt: %s\nparent of: %s' % (new_dirname, entry))
                target_parent_id = target_parent_entry[0][2]
            if target_parent_id == entry[0][2]:
                target_parent_id = None
            else:
                self.last_target_parent[0] = new_dirname
                self.last_target_parent[1] = target_parent_id
        source_exec = source_details[3]
        changed = content_change or source_parent_id != target_parent_id or old_basename != entry[0][1] or (source_exec != target_exec)
        if not changed and (not self.include_unchanged):
            return (None, False)
        else:
            if old_path is None:
                old_path = path = pathjoin(old_dirname, old_basename)
                old_path_u = self.utf8_decode(old_path, 'surrogateescape')[0]
                path_u = old_path_u
            else:
                old_path_u = self.utf8_decode(old_path, 'surrogateescape')[0]
                if old_path == path:
                    path_u = old_path_u
                else:
                    path_u = self.utf8_decode(path, 'surrogateescape')[0]
            source_kind = DirState._minikind_to_kind[source_minikind]
            return (InventoryTreeChange(entry[0][2], (old_path_u, path_u), content_change, (True, True), (source_parent_id, target_parent_id), (self.utf8_decode(old_basename, 'surrogateescape')[0], self.utf8_decode(entry[0][1], 'surrogateescape')[0]), (source_kind, target_kind), (source_exec, target_exec)), changed)
    elif source_minikind in b'a' and target_minikind in _fdlt:
        path = pathjoin(entry[0][0], entry[0][1])
        parent_id = self.state._get_entry(self.target_index, path_utf8=entry[0][0])[0][2]
        if parent_id == entry[0][2]:
            parent_id = None
        if path_info is not None:
            if self.use_filesystem_for_exec:
                target_exec = bool(stat.S_ISREG(path_info[3].st_mode) and stat.S_IEXEC & path_info[3].st_mode)
            else:
                target_exec = target_details[3]
            return (InventoryTreeChange(entry[0][2], (None, self.utf8_decode(path, 'surrogateescape')[0]), True, (False, True), (None, parent_id), (None, self.utf8_decode(entry[0][1], 'surrogateescape')[0]), (None, path_info[2]), (None, target_exec)), True)
        else:
            return (InventoryTreeChange(entry[0][2], (None, self.utf8_decode(path, 'surrogateescape')[0]), False, (False, True), (None, parent_id), (None, self.utf8_decode(entry[0][1], 'surrogateescape')[0]), (None, None), (None, False)), True)
    elif source_minikind in _fdlt and target_minikind in b'a':
        old_path = pathjoin(entry[0][0], entry[0][1])
        parent_id = self.state._get_entry(self.source_index, path_utf8=entry[0][0])[0][2]
        if parent_id == entry[0][2]:
            parent_id = None
        return (InventoryTreeChange(entry[0][2], (self.utf8_decode(old_path, 'surrogateescape')[0], None), True, (True, False), (parent_id, None), (self.utf8_decode(entry[0][1], 'surrogateescape')[0], None), (DirState._minikind_to_kind[source_minikind], None), (source_details[3], None)), True)
    elif source_minikind in _fdlt and target_minikind in b'r':
        if not osutils.is_inside_any(self.searched_specific_files, target_details[1]):
            self.search_specific_files.add(target_details[1])
    elif source_minikind in _ra and target_minikind in _ra:
        pass
    else:
        raise AssertionError("don't know how to compare source_minikind=%r, target_minikind=%r" % (source_minikind, target_minikind))
    return (None, None)