from sympy.core import S, Pow
from sympy.core.function import expand
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol, Wild
from sympy.functions import exp, sqrt, hyper
from sympy.integrals import Integral
from sympy.polys import roots, gcd
from sympy.polys.polytools import cancel, factor
from sympy.simplify import collect, simplify, logcombine # type: ignore
from sympy.simplify.powsimp import powdenest
from sympy.solvers.ode.ode import get_numbered_constants

This module contains the implementation of the 2nd_hypergeometric hint for
dsolve. This is an incomplete implementation of the algorithm described in [1].
The algorithm solves 2nd order linear ODEs of the form

.. math:: y'' + A(x) y' + B(x) y = 0\text{,}

where `A` and `B` are rational functions. The algorithm should find any
solution of the form

.. math:: y = P(x) _pF_q(..; ..;\frac{\alpha x^k + \beta}{\gamma x^k + \delta})\text{,}

where pFq is any of 2F1, 1F1 or 0F1 and `P` is an "arbitrary function".
Currently only the 2F1 case is implemented in SymPy but the other cases are
described in the paper and could be implemented in future (contributions
welcome!).

References
==========

.. [1] L. Chan, E.S. Cheb-Terrab, Non-Liouvillian solutions for second order
       linear ODEs, (2004).
       https://arxiv.org/abs/math-ph/0402063
