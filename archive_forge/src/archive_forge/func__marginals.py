import math as _math
from abc import ABCMeta, abstractmethod
from functools import reduce
@staticmethod
def _marginals(*contingency):
    """Calculates values of contingency table marginals from its values.
        QuadgramAssocMeasures._marginals(1, 0, 2, 46, 552, 825, 2577, 34967, 1, 0, 2, 48, 7250, 9031, 28585, 356653)
        (1, (2, 553, 3, 1), (7804, 6, 3132, 1378, 49, 2), (38970, 17660, 100, 38970), 440540)
        """
    n_iiii, n_oiii, n_ioii, n_ooii, n_iioi, n_oioi, n_iooi, n_oooi, n_iiio, n_oiio, n_ioio, n_ooio, n_iioo, n_oioo, n_iooo, n_oooo = contingency
    n_iiix = n_iiii + n_iiio
    n_iixi = n_iiii + n_iioi
    n_ixii = n_iiii + n_ioii
    n_xiii = n_iiii + n_oiii
    n_iixx = n_iiii + n_iioi + n_iiio + n_iioo
    n_ixix = n_iiii + n_ioii + n_iiio + n_ioio
    n_ixxi = n_iiii + n_ioii + n_iioi + n_iooi
    n_xixi = n_iiii + n_oiii + n_iioi + n_oioi
    n_xxii = n_iiii + n_oiii + n_ioii + n_ooii
    n_xiix = n_iiii + n_oiii + n_iiio + n_oiio
    n_ixxx = n_iiii + n_ioii + n_iioi + n_iiio + n_iooi + n_iioo + n_ioio + n_iooo
    n_xixx = n_iiii + n_oiii + n_iioi + n_iiio + n_oioi + n_oiio + n_iioo + n_oioo
    n_xxix = n_iiii + n_oiii + n_ioii + n_iiio + n_ooii + n_ioio + n_oiio + n_ooio
    n_xxxi = n_iiii + n_oiii + n_ioii + n_iioi + n_ooii + n_iooi + n_oioi + n_oooi
    n_all = sum(contingency)
    return (n_iiii, (n_iiix, n_iixi, n_ixii, n_xiii), (n_iixx, n_ixix, n_ixxi, n_xixi, n_xxii, n_xiix), (n_ixxx, n_xixx, n_xxix, n_xxxi), n_all)