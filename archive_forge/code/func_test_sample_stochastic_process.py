from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple
from sympy.core.function import Lambda
from sympy.core.numbers import (Float, Rational, oo, pi)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import (gamma, lowergamma)
from sympy.logic.boolalg import (And, Not)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.immutable import ImmutableMatrix
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (FiniteSet, Interval)
from sympy.stats import (DiscreteMarkovChain, P, TransitionMatrixOf, E,
from sympy.stats.joint_rv import JointDistribution
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.rv import RandomIndexedSymbol
from sympy.stats.symbolic_probability import Probability, Expectation
from sympy.testing.pytest import (raises, skip, ignore_warnings,
from sympy.external import import_module
from sympy.stats.frv_types import BernoulliDistribution
from sympy.stats.drv_types import PoissonDistribution
from sympy.stats.crv_types import NormalDistribution, GammaDistribution
from sympy.core.symbol import Str
def test_sample_stochastic_process():
    if not import_module('scipy'):
        skip('SciPy Not installed. Skip sampling tests')
    import random
    random.seed(0)
    numpy = import_module('numpy')
    if numpy:
        numpy.random.seed(0)
    T = Matrix([[0.5, 0.2, 0.3], [0.2, 0.5, 0.3], [0.2, 0.3, 0.5]])
    Y = DiscreteMarkovChain('Y', [0, 1, 2], T)
    for samps in range(10):
        assert next(sample_stochastic_process(Y)) in Y.state_space
    Z = DiscreteMarkovChain('Z', ['1', 1, 0], T)
    for samps in range(10):
        assert next(sample_stochastic_process(Z)) in Z.state_space
    T = Matrix([[S.Half, Rational(1, 4), Rational(1, 4)], [Rational(1, 3), 0, Rational(2, 3)], [S.Half, S.Half, 0]])
    X = DiscreteMarkovChain('X', [0, 1, 2], T)
    for samps in range(10):
        assert next(sample_stochastic_process(X)) in X.state_space
    W = DiscreteMarkovChain('W', [1, pi, oo], T)
    for samps in range(10):
        assert next(sample_stochastic_process(W)) in W.state_space