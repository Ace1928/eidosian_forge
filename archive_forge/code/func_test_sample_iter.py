from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.function import Lambda
from sympy.core.numbers import (Rational, nan, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import (FallingFactorial, binomial)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.delta_functions import DiracDelta
from sympy.integrals.integrals import integrate
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.dense import Matrix
from sympy.sets.sets import Interval
from sympy.tensor.indexed import Indexed
from sympy.stats import (Die, Normal, Exponential, FiniteRV, P, E, H, variance,
from sympy.stats.rv import (IndependentProductPSpace, rs_swap, Density, NamedArgsMixin,
from sympy.testing.pytest import raises, skip, XFAIL, warns_deprecated_sympy
from sympy.external import import_module
from sympy.core.numbers import comp
from sympy.stats.frv_types import BernoulliDistribution
from sympy.core.symbol import Dummy
from sympy.functions.elementary.piecewise import Piecewise
def test_sample_iter():
    X = Normal('X', 0, 1)
    Y = DiscreteUniform('Y', [1, 2, 7])
    Z = Poisson('Z', 2)
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy is not installed. Abort tests')
    expr = X ** 2 + 3
    iterator = sample_iter(expr)
    expr2 = Y ** 2 + 5 * Y + 4
    iterator2 = sample_iter(expr2)
    expr3 = Z ** 3 + 4
    iterator3 = sample_iter(expr3)

    def is_iterator(obj):
        if hasattr(obj, '__iter__') and (hasattr(obj, 'next') or hasattr(obj, '__next__')) and callable(obj.__iter__) and (obj.__iter__() is obj):
            return True
        else:
            return False
    assert is_iterator(iterator)
    assert is_iterator(iterator2)
    assert is_iterator(iterator3)