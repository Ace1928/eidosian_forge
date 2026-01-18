from ...sage_helper import _within_sage, sage_method
from .extended_bloch import *
from ...snap import t3mlite as t3m
@sage_method
def verified_complex_volume_from_lifted_ptolemys(mcomplex, ptolemys):
    """
    Given lifted Ptolemy coordinates for a triangulation (as dictionary)
    and the number of tetrahedra, compute the complex volume (where
    the real part is the Chern-Simons and the imaginary part is the
    volume).

    The result is correct modulo pi^2/2.
    """
    result = compute_complex_volume_from_lifted_ptolemys_no_torsion_adjustment(len(mcomplex.Tetrahedra), ptolemys)
    CIF = result.parent()
    return result + _compute_adjustment(mcomplex) * CIF(pi ** 2 / 6)