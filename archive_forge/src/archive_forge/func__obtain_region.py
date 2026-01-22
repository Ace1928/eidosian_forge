from .util import (
import sys
from functools import reduce
def _obtain_region(self, a, offset, size, flags, is_recursive):
    r = None
    lo = 0
    hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        ofs = a[mid]._b
        if ofs <= offset:
            if a[mid].includes_ofs(offset):
                r = a[mid]
                break
            lo = mid + 1
        else:
            hi = mid
    if r is None:
        window_size = self._window_size
        left = self.MapWindowCls(0, 0)
        mid = self.MapWindowCls(offset, size)
        right = self.MapWindowCls(a.file_size(), 0)
        if self._memory_size + window_size > self._max_memory_size:
            self._collect_lru_region(window_size)
        insert_pos = 0
        len_regions = len(a)
        if len_regions == 1:
            if a[0]._b <= offset:
                insert_pos = 1
        else:
            insert_pos = len_regions
            for i, region in enumerate(a):
                if region._b > offset:
                    insert_pos = i
                    break
        if insert_pos == 0:
            if len_regions:
                right = self.MapWindowCls.from_region(a[insert_pos])
        else:
            if insert_pos != len_regions:
                right = self.MapWindowCls.from_region(a[insert_pos])
            left = self.MapWindowCls.from_region(a[insert_pos - 1])
        mid.extend_left_to(left, window_size)
        mid.extend_right_to(right, window_size)
        mid.align()
        if mid.ofs_end() > right.ofs:
            mid.size = right.ofs - mid.ofs
        try:
            if self._handle_count >= self._max_handle_count:
                raise Exception
            r = self.MapRegionCls(a.path_or_fd(), mid.ofs, mid.size, flags)
        except Exception:
            if is_recursive:
                raise
            self._collect_lru_region(0)
            return self._obtain_region(a, offset, size, flags, True)
        self._handle_count += 1
        self._memory_size += r.size()
        a.insert(insert_pos, r)
    return r