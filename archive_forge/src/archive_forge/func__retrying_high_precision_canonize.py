from ..sage_helper import _within_sage, sage_method
from .cuspCrossSection import RealCuspCrossSection
from .squareExtensions import find_shapes_as_complex_sqrt_lin_combinations
from . import verifyHyperbolicity
from . import exceptions
from ..exceptions import SnapPeaFatalError
from ..snap import t3mlite as t3m
def _retrying_high_precision_canonize(M):
    """
    Wrapper for SnapPea kernel's function to compute the proto-canonical
    triangulation. It will retry the kernel function if it fails, switching
    to the quad-double implementation.
    Returns the proto-canonical triangulation if the kernel function was
    successful eventually. Otherwise None. The original manifold is unchanged.
    """
    Mcopy = M.copy()
    if _retrying_canonize(Mcopy):
        return Mcopy
    Mhp = M.high_precision()
    if _retrying_canonize(Mhp):
        return Mhp
    return None