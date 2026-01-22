from ...sage_helper import _within_sage, sage_method
from ...math_basics import prod
from ...snap import peripheral
from .adjust_torsion import *
from .compute_ptolemys import *
from .. import verifyHyperbolicity
from ..cuspCrossSection import ComplexCuspCrossSection
from ...snap import t3mlite as t3m
def _compute_holonomy(manifold, shapes):
    """
    Computes the holonomy for the peripheral curves for the given 1-cusped
    manifold and shape intervals.
    """
    zp = [1 / (1 - z) for z in shapes]
    zpp = [(z - 1) / z for z in shapes]
    cross_ratios = [z for triple in zip(shapes, zp, zpp) for z in triple]
    trig = manifold.without_hyperbolic_structure()
    trig.dehn_fill((0, 0))
    peripheral_eqns = trig.gluing_equations()[-2:]
    return [prod([l ** expo for l, expo in zip(cross_ratios, eqn)]) for eqn in peripheral_eqns]