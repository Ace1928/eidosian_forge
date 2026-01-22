from heapq import heappop, heappush
from .lru_cache import LRUCache
def _has_candidates(wlst, cstates):
    for dt, cmt in wlst.iter():
        if cmt in cstates:
            if not cstates[cmt] & _DNC == _DNC:
                return True
    return False