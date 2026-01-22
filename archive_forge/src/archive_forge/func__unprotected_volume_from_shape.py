from ..sage_helper import sage_method, _within_sage
from ..number import Number
from . import verifyHyperbolicity
def _unprotected_volume_from_shape(z):
    """
    Computes the Bloch-Wigner dilogarithm for z assuming z is of a type that
    properly supports polylog.
    """
    return (1 - z).arg() * z.abs().log() + z.polylog(2).imag()