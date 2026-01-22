from .line import R13LineWithMatrix
from ..verify.shapes import compute_hyperbolic_shapes # type: ignore
from ..snap.fundamental_polyhedron import FundamentalPolyhedronEngine # type: ignore
from ..snap.kernel_structures import TransferKernelStructuresEngine # type: ignore
from ..snap.t3mlite import simplex, Mcomplex, Tetrahedron, Vertex # type: ignore
from ..SnapPy import word_as_list # type: ignore
from ..hyperboloid import (o13_inverse,  # type: ignore
from ..upper_halfspace import sl2c_inverse, psl2c_to_o13 # type: ignore
from ..upper_halfspace.ideal_point import ideal_point_to_r13 # type: ignore
from ..matrix import vector, matrix, mat_solve # type: ignore
from ..math_basics import prod, xgcd # type: ignore
from collections import deque
from typing import Tuple, Sequence, Optional, Any
def add_r13_geometry(mcomplex: Mcomplex, manifold, verified: bool=False, bits_prec: Optional[int]=None):
    """
    Given the same triangulation once as Mcomplex and once as SnapPy Manifold,
    develops the vertices of the tetrahedra (using the same fundamental
    polyhedron as the SnapPea kernel), computes the face-pairing matrices and
    the matrices corresponding to the generators of the unsimplified
    fundamental group, computes the incenter of the base tetrahedron and
    the core curve for each vertex of each tetrahedron corresponding to a
    filled cusp.

    The precision can be given by bits_prec (if not given, the precision of
    the Manifold type is used, i.e., 53 for Manifold and 212 for ManifoldHP).

    If verified is True, intervals will be computed for all the above
    information.
    """
    shapes = compute_hyperbolic_shapes(manifold, verified=verified, bits_prec=bits_prec)
    z = shapes[0]
    RF = z.real().parent()
    poly = FundamentalPolyhedronEngine.from_manifold_and_shapes(manifold, shapes, normalize_matrices=True)
    TransferKernelStructuresEngine(mcomplex, manifold).reindex_cusps_and_transfer_peripheral_curves()
    mcomplex.verified = verified
    mcomplex.RF = RF
    mcomplex.GeneratorMatrices = {g: _to_matrix(m) for g, m in poly.mcomplex.GeneratorMatrices.items()}
    mcomplex.num_generators = len(mcomplex.GeneratorMatrices) // 2
    for tet, developed_tet in zip(mcomplex.Tetrahedra, poly.mcomplex):
        tet.ShapeParameters = developed_tet.ShapeParameters
        tet.ideal_vertices = {V: developed_tet.Class[V].IdealPoint for V in simplex.ZeroSubsimplices}
        tet.R13_vertices = {V: ideal_point_to_r13(z, RF) for V, z in tet.ideal_vertices.items()}
        compute_r13_planes_for_tet(tet)
        tet.O13_matrices = {F: psl2c_to_o13(mcomplex.GeneratorMatrices.get(-g)) for F, g in developed_tet.GeneratorsInfo.items()}
        tet.core_curves = {}
    mcomplex.baseTet = mcomplex.Tetrahedra[poly.mcomplex.ChooseGenInitialTet.Index]
    mcomplex.baseTetInRadius, mcomplex.R13_baseTetInCenter = _compute_inradius_and_incenter_from_planes([mcomplex.baseTet.R13_planes[f] for f in simplex.TwoSubsimplices])
    all_peripheral_words: Optional[Sequence[Sequence[Sequence[int]]]] = None
    for v, info in zip(mcomplex.Vertices, manifold.cusp_info()):
        v.filling_matrix = _filling_matrix(info)
        if v.filling_matrix[0] != (0, 0):
            if all_peripheral_words is None:
                G = manifold.fundamental_group(False)
                all_peripheral_words = G.peripheral_curves(as_int_list=True)
            _develop_core_curve_cusp(mcomplex, v, _compute_core_curve(mcomplex, all_peripheral_words[v.Index], v.filling_matrix[1]))
    return mcomplex