from warnings import warn as _warn
from math import log as _log, exp as _exp, pi as _pi, e as _e, ceil as _ceil
from math import sqrt as _sqrt, acos as _acos, cos as _cos, sin as _sin
from math import tau as TWOPI, floor as _floor, isfinite as _isfinite
from os import urandom as _urandom
from _collections_abc import Set as _Set, Sequence as _Sequence
from operator import index as _index
from itertools import accumulate as _accumulate, repeat as _repeat
from bisect import bisect as _bisect
import os as _os
import _random
def vonmisesvariate(self, mu, kappa):
    """Circular data distribution.

        mu is the mean angle, expressed in radians between 0 and 2*pi, and
        kappa is the concentration parameter, which must be greater than or
        equal to zero.  If kappa is equal to zero, this distribution reduces
        to a uniform random angle over the range 0 to 2*pi.

        """
    random = self.random
    if kappa <= 1e-06:
        return TWOPI * random()
    s = 0.5 / kappa
    r = s + _sqrt(1.0 + s * s)
    while True:
        u1 = random()
        z = _cos(_pi * u1)
        d = z / (r + z)
        u2 = random()
        if u2 < 1.0 - d * d or u2 <= (1.0 - d) * _exp(d):
            break
    q = 1.0 / r
    f = (q + z) / (1.0 + q * z)
    u3 = random()
    if u3 > 0.5:
        theta = (mu + _acos(f)) % TWOPI
    else:
        theta = (mu - _acos(f)) % TWOPI
    return theta