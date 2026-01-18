from ..sage_helper import _within_sage, sage_method
from .. import snap
from . import exceptions

    Given an orientable SnapPy Manifold, verifies its hyperbolicity.

    Similar to HIKMOT's :py:meth:`verify_hyperbolicity`, the result is either
    ``(True, listOfShapeIntervals)`` or ``(False, [])`` if verification failed.
    ``listOfShapesIntervals`` is a list of complex intervals (elements in
    sage's ``ComplexIntervalField``) certified to contain the true shapes
    for the hyperbolic manifold.

    Higher precision intervals can be obtained by setting ``bits_prec``::

        sage: from snappy import Manifold
        sage: M = Manifold("m019")
        sage: M.verify_hyperbolicity() # doctest: +NUMERIC12
        (True, [0.780552527850? + 0.914473662967?*I, 0.780552527850? + 0.91447366296773?*I, 0.4600211755737? + 0.6326241936052?*I])

        sage: M = Manifold("t02333(3,4)")
        sage: M.verify_hyperbolicity() # doctest: +NUMERIC9
        (True, [2.152188153612? + 0.284940667895?*I, 1.92308491369? + 1.10360701507?*I, 0.014388591584? + 0.143084469681?*I, -2.5493670288? + 3.7453498408?*I, 0.142120333822? + 0.176540027036?*I, 0.504866865874? + 0.82829881681?*I, 0.50479249917? + 0.98036162786?*I, -0.589495705074? + 0.81267480427?*I])

    One can instead get a holonomy representation associated to the
    verified hyperbolic structure.  This representation takes values
    in 2x2 matrices with entries in the ``ComplexIntervalField``::

        sage: M = Manifold("m004(1,2)")
        sage: success, rho = M.verify_hyperbolicity(holonomy=True)
        sage: success
        True
        sage: trace = rho('aaB').trace(); trace # doctest: +NUMERIC9
        -0.1118628555? + 3.8536121048?*I
        sage: (trace - 2).contains_zero()
        False
        sage: (rho('aBAbaabAB').trace() - 2).contains_zero()
        True

    Here, there is **provably** a fixed holonomy representation rho0
    from the fundamental group G of M to SL(2, C) so that for each
    element g of G the matrix rho0(g) is contained in rho(g).  In
    particular, the above constitutes a proof that the word 'aaB' is
    non-trivial in G.  In contrast, the final computation is
    consistent with 'aBAbaabAB' being trivial in G, but *does not prove
    this*.

    A non-hyperbolic manifold (``False`` indicates that the manifold
    might not be hyperbolic but does **not** certify
    non-hyperbolicity. Sometimes, hyperbolicity can only be verified
    after increasing the precision)::

        sage: M = Manifold("4_1(1,0)")
        sage: M.verify_hyperbolicity()
        (False, [])

    Under the hood, the function will call the ``CertifiedShapesEngine`` to produce
    intervals certified to contain a solution to the rectangular gluing equations.
    It then calls ``check_logarithmic_gluing_equations_and_positively_oriented_tets``
    to verify that the logarithmic gluing equations are fulfilled and that all
    tetrahedra are positively oriented.
    