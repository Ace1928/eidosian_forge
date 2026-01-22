from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def _testing_check_against_snappea(self, epsilon):
    """
        Compare the computed edge lengths and tilts against the one computed by
        the SnapPea kernel.

        >>> from snappy import Manifold

        Convention of the kernel is to use (3/8) sqrt(3) as area (ensuring that
        cusp neighborhoods are disjoint).

        >>> cusp_area = 0.649519052838329

        >>> for name in ['m009', 'm015', 't02333']:
        ...     M = Manifold(name)
        ...     e = RealCuspCrossSection.fromManifoldAndShapes(M, M.tetrahedra_shapes('rect'))
        ...     e.normalize_cusps(cusp_area)
        ...     e._testing_check_against_snappea(1e-10)

        """
    CuspCrossSectionBase._testing_check_against_snappea(self, epsilon)
    TwoSubs = t3m.simplex.TwoSubsimplices
    snappea_tilts, snappea_edges = self.manifold._cusp_cross_section_info()
    for tet, snappea_tet_tilts in zip(self.mcomplex.Tetrahedra, snappea_tilts):
        for f, snappea_tet_tilt in zip(TwoSubs, snappea_tet_tilts):
            tilt = RealCuspCrossSection._tet_tilt(tet, f)
            if not abs(snappea_tet_tilt - tilt) < epsilon:
                raise ConsistencyWithSnapPeaNumericalVerifyError(snappea_tet_tilt, tilt)