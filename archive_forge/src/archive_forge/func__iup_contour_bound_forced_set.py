from typing import (
from numbers import Integral, Real
@cython.locals(cj=cython.double, dj=cython.double, lcj=cython.double, ldj=cython.double, ncj=cython.double, ndj=cython.double, force=cython.int, forced=set)
def _iup_contour_bound_forced_set(deltas: _DeltaSegment, coords: _PointSegment, tolerance: Real=0) -> set:
    """The forced set is a conservative set of points on the contour that must be encoded
    explicitly (ie. cannot be interpolated).  Calculating this set allows for significantly
    speeding up the dynamic-programming, as well as resolve circularity in DP.

    The set is precise; that is, if an index is in the returned set, then there is no way
    that IUP can generate delta for that point, given `coords` and `deltas`.
    """
    assert len(deltas) == len(coords)
    n = len(deltas)
    forced = set()
    for i in range(len(deltas) - 1, -1, -1):
        ld, lc = (deltas[i - 1], coords[i - 1])
        d, c = (deltas[i], coords[i])
        nd, nc = (deltas[i - n + 1], coords[i - n + 1])
        for j in (0, 1):
            cj = c[j]
            dj = d[j]
            lcj = lc[j]
            ldj = ld[j]
            ncj = nc[j]
            ndj = nd[j]
            if lcj <= ncj:
                c1, c2 = (lcj, ncj)
                d1, d2 = (ldj, ndj)
            else:
                c1, c2 = (ncj, lcj)
                d1, d2 = (ndj, ldj)
            force = False
            if c1 == c2:
                if abs(d1 - d2) > tolerance and abs(dj) > tolerance:
                    force = True
            elif c1 <= cj <= c2:
                if not min(d1, d2) - tolerance <= dj <= max(d1, d2) + tolerance:
                    force = True
            elif d1 != d2:
                if cj < c1:
                    if abs(dj) > tolerance and abs(dj - d1) > tolerance and ((dj - tolerance < d1) != (d1 < d2)):
                        force = True
                elif abs(dj) > tolerance and abs(dj - d2) > tolerance and ((d2 < dj + tolerance) != (d1 < d2)):
                    force = True
            if force:
                forced.add(i)
                break
    return forced