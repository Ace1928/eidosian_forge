from ...sage_helper import _within_sage
from ..upper_halfspace.finite_point import *


        sage: from sage.all import *
        sage: t0 = CIF(RIF(2.3, 2.30000000001), 3.4)
        sage: t1 = CIF(4.32, RIF(5.43, 5.4300000001))
        sage: c = CuspTranslateEngine(t0, t1)
        sage: z = CIF(RIF(0.23, 0.26), 0.43)
        sage: perturb = CIF(0.01, 0)
        sage: t = RIF(5)

        sage: for i in range(-2, 3): # doctest: +NUMERIC6
        ...     for j in range(-2, 3):
        ...         print(c.translate_to_match(FinitePoint(z + i * t0 + j * t1 + perturb, t), FinitePoint(z, t)))
        FinitePoint(0.3? + 0.430000000?*I, 5)
        FinitePoint(0.3? + 0.430000000?*I, 5)
        FinitePoint(0.3? + 0.430000000000000?*I, 5)
        FinitePoint(0.3? + 0.430000000?*I, 5)
        FinitePoint(0.3? + 0.430000000?*I, 5)
        FinitePoint(0.3? + 0.430000000?*I, 5)
        FinitePoint(0.3? + 0.430000000?*I, 5)
        FinitePoint(0.3? + 0.430000000000000?*I, 5)
        FinitePoint(0.3? + 0.430000000?*I, 5)
        FinitePoint(0.3? + 0.430000000?*I, 5)
        FinitePoint(0.3? + 0.430000000?*I, 5)
        FinitePoint(0.3? + 0.430000000?*I, 5)
        FinitePoint(0.3? + 0.43000000000000000?*I, 5)
        FinitePoint(0.3? + 0.430000000?*I, 5)
        FinitePoint(0.3? + 0.430000000?*I, 5)
        FinitePoint(0.3? + 0.430000000?*I, 5)
        FinitePoint(0.3? + 0.430000000?*I, 5)
        FinitePoint(0.3? + 0.430000000000000?*I, 5)
        FinitePoint(0.3? + 0.430000000?*I, 5)
        FinitePoint(0.3? + 0.430000000?*I, 5)
        FinitePoint(0.3? + 0.430000000?*I, 5)
        FinitePoint(0.3? + 0.430000000?*I, 5)
        FinitePoint(0.3? + 0.430000000000000?*I, 5)
        FinitePoint(0.3? + 0.430000000?*I, 5)
        FinitePoint(0.3? + 0.430000000?*I, 5)

        sage: perturb = CIF(0.1, 0)
        sage: c.translate_to_match(FinitePoint(z + 2 * t0 + 1 * t1 + perturb, t), FinitePoint(z, t)) is None
        True
        