import numpy as np
from ._ufuncs import _ellip_harm
from ._ellip_harm_2 import _ellipsoid, _ellipsoid_norm
def _ellip_normal_vec(h2, k2, n, p):
    return _ellipsoid_norm(h2, k2, n, p)