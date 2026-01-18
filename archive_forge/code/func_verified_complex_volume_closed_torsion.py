from ...sage_helper import _within_sage, sage_method
from ...math_basics import prod
from ...snap import peripheral
from .adjust_torsion import *
from .compute_ptolemys import *
from .. import verifyHyperbolicity
from ..cuspCrossSection import ComplexCuspCrossSection
from ...snap import t3mlite as t3m
@sage_method
def verified_complex_volume_closed_torsion(manifold, bits_prec=None):
    """
    Computes the verified complex volume (where the real part is the
    volume and the imaginary part is the Chern-Simons) for a given
    SnapPy.Manifold.

    Note that the result is correct only up to two torsion, i.e.,
    up to multiples of pi^2/2. The method expects an oriented manifold
    with exactly one cusp which is filled, otherwise it raises an exception.

    If bits_prec is unspecified, the default precision of
    SnapPy.Manifold or SnapPy.ManifoldHP, respectively, will be used.
    """
    if manifold.num_cusps() != 1:
        raise ValueError('The method does not support the given manifold because it does not have exactly one cusp.')
    if manifold.cusp_info()[0]['complete?']:
        raise ValueError('The method does not support the given manifold because it is not a closed manifold.')
    shapes = manifold.tetrahedra_shapes('rect', bits_prec=bits_prec, intervals=True)
    verifyHyperbolicity.check_logarithmic_gluing_equations_and_positively_oriented_tets(manifold, shapes)
    m_holonomy, l_holonomy = _compute_holonomy(manifold, shapes)
    m_star, l_star = peripheral.peripheral_cohomology_basis(manifold)
    cusp_dual_edges = [(i, F, V) for i in range(manifold.num_tetrahedra()) for F in t3m.TwoSubsimplices for V in t3m.ZeroSubsimplices if F & V]
    one_cocycle = {k: 1 / (m_holonomy ** m_star[k] * l_holonomy ** l_star[k]) for k in cusp_dual_edges}
    c = ComplexCuspCrossSection.fromManifoldAndShapes(manifold, shapes, one_cocycle)
    m_lifted_holonomy, l_lifted_holonomy = zero_lifted_holonomy(manifold, m_holonomy.log() / 2, l_holonomy.log() / 2, 1)
    lifted_one_cocycle = {k: m_lifted_holonomy * m_star[k] + l_lifted_holonomy * l_star[k] for k in cusp_dual_edges}
    lifted_ptolemys = lifted_ptolemys_from_cross_section(c, lifted_one_cocycle)
    complex_volume = verified_complex_volume_from_lifted_ptolemys(c.mcomplex, lifted_ptolemys)
    return normalize_by_pi_square_over_two(complex_volume) / sage.all.I