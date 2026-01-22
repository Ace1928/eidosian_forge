import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import check_nD, deprecate_func
from ..util import crop
from ._skeletonize_3d_cy import _compute_thin_image
from ._skeletonize_cy import _fast_skeletonize, _skeletonize_loop, _table_lookup_index
def _generate_thin_luts():
    """generate LUTs for thinning algorithm (for reference)"""

    def nabe(n):
        return np.array([n >> i & 1 for i in range(0, 9)]).astype(bool)

    def G1(n):
        s = 0
        bits = nabe(n)
        for i in (0, 2, 4, 6):
            if not bits[i] and (bits[i + 1] or bits[(i + 2) % 8]):
                s += 1
        return s == 1
    g1_lut = np.array([G1(n) for n in range(256)])

    def G2(n):
        n1, n2 = (0, 0)
        bits = nabe(n)
        for k in (1, 3, 5, 7):
            if bits[k] or bits[k - 1]:
                n1 += 1
            if bits[k] or bits[(k + 1) % 8]:
                n2 += 1
        return min(n1, n2) in [2, 3]
    g2_lut = np.array([G2(n) for n in range(256)])
    g12_lut = g1_lut & g2_lut

    def G3(n):
        bits = nabe(n)
        return not ((bits[1] or bits[2] or (not bits[7])) and bits[0])

    def G3p(n):
        bits = nabe(n)
        return not ((bits[5] or bits[6] or (not bits[3])) and bits[4])
    g3_lut = np.array([G3(n) for n in range(256)])
    g3p_lut = np.array([G3p(n) for n in range(256)])
    g123_lut = g12_lut & g3_lut
    g123p_lut = g12_lut & g3p_lut
    return (g123_lut, g123p_lut)