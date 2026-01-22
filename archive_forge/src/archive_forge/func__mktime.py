import time as _time
import math as _math
import sys
from operator import index as _index
def _mktime(self):
    """Return integer POSIX timestamp."""
    epoch = datetime(1970, 1, 1)
    max_fold_seconds = 24 * 3600
    t = (self - epoch) // timedelta(0, 1)

    def local(u):
        y, m, d, hh, mm, ss = _time.localtime(u)[:6]
        return (datetime(y, m, d, hh, mm, ss) - epoch) // timedelta(0, 1)
    a = local(t) - t
    u1 = t - a
    t1 = local(u1)
    if t1 == t:
        u2 = u1 + (-max_fold_seconds, max_fold_seconds)[self.fold]
        b = local(u2) - u2
        if a == b:
            return u1
    else:
        b = t1 - u1
        assert a != b
    u2 = t - b
    t2 = local(u2)
    if t2 == t:
        return u2
    if t1 == t:
        return u1
    return (max, min)[self.fold](u1, u2)