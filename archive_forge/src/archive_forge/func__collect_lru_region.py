from .util import (
import sys
from functools import reduce
def _collect_lru_region(self, size):
    """Unmap the region which was least-recently used and has no client
        :param size: size of the region we want to map next (assuming its not already mapped partially or full
            if 0, we try to free any available region
        :return: Amount of freed regions

        .. Note::
            We don't raise exceptions anymore, in order to keep the system working, allowing temporary overallocation.
            If the system runs out of memory, it will tell.

        .. TODO::
            implement a case where all unusued regions are discarded efficiently.
            Currently its only brute force
        """
    num_found = 0
    while size == 0 or self._memory_size + size > self._max_memory_size:
        lru_region = None
        lru_list = None
        for regions in self._fdict.values():
            for region in regions:
                if region.client_count() == 1 and (lru_region is None or region._uc < lru_region._uc):
                    lru_region = region
                    lru_list = regions
        if lru_region is None:
            break
        num_found += 1
        del lru_list[lru_list.index(lru_region)]
        lru_region.increment_client_count(-1)
        self._memory_size -= lru_region.size()
        self._handle_count -= 1
    return num_found