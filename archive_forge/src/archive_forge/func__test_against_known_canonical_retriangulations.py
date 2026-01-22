from ..sage_helper import _within_sage, sage_method
from .cuspCrossSection import RealCuspCrossSection
from .squareExtensions import find_shapes_as_complex_sqrt_lin_combinations
from . import verifyHyperbolicity
from . import exceptions
from ..exceptions import SnapPeaFatalError
from ..snap import t3mlite as t3m
def _test_against_known_canonical_retriangulations():
    from snappy import Manifold
    for name, bytes_ in _known_canonical_retriangulations:
        M = Manifold(name)
        K = verified_canonical_retriangulation(M)
        L = Manifold('empty')
        L._from_bytes(bytes_)
        if not K.isomorphisms_to(L):
            raise Exception('%s failed' % name)