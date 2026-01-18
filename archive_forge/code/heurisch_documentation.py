from __future__ import annotations
from itertools import permutations
from functools import reduce
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.symbol import Wild, Dummy, Symbol
from sympy.core.basic import sympify
from sympy.core.numbers import Rational, pi, I
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.traversal import iterfreeargs
from sympy.functions import exp, sin, cos, tan, cot, asin, atan
from sympy.functions import log, sinh, cosh, tanh, coth, asinh
from sympy.functions import sqrt, erf, erfi, li, Ei
from sympy.functions import besselj, bessely, besseli, besselk
from sympy.functions import hankel1, hankel2, jn, yn
from sympy.functions.elementary.complexes import Abs, re, im, sign, arg
from sympy.functions.elementary.exponential import LambertW
from sympy.functions.elementary.integers import floor, ceiling
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import Heaviside, DiracDelta
from sympy.simplify.radsimp import collect
from sympy.logic.boolalg import And, Or
from sympy.utilities.iterables import uniq
from sympy.polys import quo, gcd, lcm, factor_list, cancel, PolynomialError
from sympy.polys.monomials import itermonomials
from sympy.polys.polyroots import root_factors
from sympy.polys.rings import PolyRing
from sympy.polys.solvers import solve_lin_sys
from sympy.polys.constructor import construct_domain
from sympy.integrals.integrals import integrate

    Compute indefinite integral using heuristic Risch algorithm.

    Explanation
    ===========

    This is a heuristic approach to indefinite integration in finite
    terms using the extended heuristic (parallel) Risch algorithm, based
    on Manuel Bronstein's "Poor Man's Integrator".

    The algorithm supports various classes of functions including
    transcendental elementary or special functions like Airy,
    Bessel, Whittaker and Lambert.

    Note that this algorithm is not a decision procedure. If it isn't
    able to compute the antiderivative for a given function, then this is
    not a proof that such a functions does not exist.  One should use
    recursive Risch algorithm in such case.  It's an open question if
    this algorithm can be made a full decision procedure.

    This is an internal integrator procedure. You should use top level
    'integrate' function in most cases, as this procedure needs some
    preprocessing steps and otherwise may fail.

    Specification
    =============

     heurisch(f, x, rewrite=False, hints=None)

       where
         f : expression
         x : symbol

         rewrite -> force rewrite 'f' in terms of 'tan' and 'tanh'
         hints   -> a list of functions that may appear in anti-derivate

          - hints = None          --> no suggestions at all
          - hints = [ ]           --> try to figure out
          - hints = [f1, ..., fn] --> we know better

    Examples
    ========

    >>> from sympy import tan
    >>> from sympy.integrals.heurisch import heurisch
    >>> from sympy.abc import x, y

    >>> heurisch(y*tan(x), x)
    y*log(tan(x)**2 + 1)/2

    See Manuel Bronstein's "Poor Man's Integrator":

    References
    ==========

    .. [1] https://www-sop.inria.fr/cafe/Manuel.Bronstein/pmint/index.html

    For more information on the implemented algorithm refer to:

    .. [2] K. Geddes, L. Stefanus, On the Risch-Norman Integration
       Method and its Implementation in Maple, Proceedings of
       ISSAC'89, ACM Press, 212-217.

    .. [3] J. H. Davenport, On the Parallel Risch Algorithm (I),
       Proceedings of EUROCAM'82, LNCS 144, Springer, 144-157.

    .. [4] J. H. Davenport, On the Parallel Risch Algorithm (III):
       Use of Tangents, SIGSAM Bulletin 16 (1982), 3-6.

    .. [5] J. H. Davenport, B. M. Trager, On the Parallel Risch
       Algorithm (II), ACM Transactions on Mathematical
       Software 11 (1985), 356-362.

    See Also
    ========

    sympy.integrals.integrals.Integral.doit
    sympy.integrals.integrals.Integral
    sympy.integrals.heurisch.components
    