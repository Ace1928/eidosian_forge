from ..sage_helper import _within_sage, sage_method
from .cuspCrossSection import RealCuspCrossSection
from .squareExtensions import find_shapes_as_complex_sqrt_lin_combinations
from . import verifyHyperbolicity
from . import exceptions
from ..exceptions import SnapPeaFatalError
from ..snap import t3mlite as t3m
def _retrying_canonize(M):
    """
    Wrapper for SnapPea kernel's function to compute the proto-canonical
    triangulation in place. It will retry the kernel function if it fails.
    Returns True if and only if the kernel function was successful eventually.
    """
    for i in range(_num_tries_canonize):
        try:
            M.canonize()
            return True
        except (RuntimeError, SnapPeaFatalError):
            M.randomize()
    return False