from ..drilling import compute_geodesic_info
from ..drilling.geodesic_tube import GeodesicTube
from ..drilling.line import distance_r13_lines
from ..snap.t3mlite import simplex # type: ignore
def _normalize_complex_length(z):
    imag = z.imag()
    CF = z.parent()
    RF = imag.parent()
    two_pi = RF('6.283185307179586476925286766559005768394338798750')
    I = CF('I')
    n = (imag / two_pi - RF('0.00000001')).round()
    return z - n * two_pi * I