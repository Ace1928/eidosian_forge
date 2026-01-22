from heapq import heappop, heappush
from .lru_cache import LRUCache
def _find_lcas(lookup_parents, c1, c2s, lookup_stamp, min_stamp=0):
    cands = []
    cstates = {}
    _ANC_OF_1 = 1
    _ANC_OF_2 = 2
    _DNC = 4
    _LCA = 8

    def _has_candidates(wlst, cstates):
        for dt, cmt in wlst.iter():
            if cmt in cstates:
                if not cstates[cmt] & _DNC == _DNC:
                    return True
        return False
    wlst = WorkList()
    cstates[c1] = _ANC_OF_1
    wlst.add((lookup_stamp(c1), c1))
    for c2 in c2s:
        cflags = cstates.get(c2, 0)
        cstates[c2] = cflags | _ANC_OF_2
        wlst.add((lookup_stamp(c2), c2))
    while _has_candidates(wlst, cstates):
        dt, cmt = wlst.get()
        cflags = cstates[cmt] & (_ANC_OF_1 | _ANC_OF_2 | _DNC)
        if cflags == _ANC_OF_1 | _ANC_OF_2:
            if not cstates[cmt] & _LCA == _LCA:
                cstates[cmt] = cstates[cmt] | _LCA
                cands.append((dt, cmt))
            cflags = cflags | _DNC
        parents = lookup_parents(cmt)
        if parents:
            for pcmt in parents:
                pflags = cstates.get(pcmt, 0)
                if pflags & cflags == cflags:
                    continue
                pdt = lookup_stamp(pcmt)
                if pdt < min_stamp:
                    continue
                cstates[pcmt] = pflags | cflags
                wlst.add((pdt, pcmt))
    results = []
    for dt, cmt in cands:
        if not cstates[cmt] & _DNC == _DNC and (dt, cmt) not in results:
            results.append((dt, cmt))
    results.sort(key=lambda x: x[0])
    lcas = [cmt for dt, cmt in results]
    return lcas