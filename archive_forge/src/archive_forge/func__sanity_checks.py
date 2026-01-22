import random
import itertools
from typing import (Sequence as tSequence, Union as tUnion, List as tList,
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda)
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, Rational, igcd, oo, pi)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.gamma_functions import gamma
from sympy.logic.boolalg import (And, Not, Or)
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.dense import (Matrix, eye, ones, zeros)
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.matrices.immutable import ImmutableMatrix
from sympy.sets.conditionset import ConditionSet
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (FiniteSet, Intersection, Interval, Set, Union)
from sympy.solvers.solveset import linsolve
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import strongly_connected_components
from sympy.stats.joint_rv import JointDistribution
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.rv import (RandomIndexedSymbol, random_symbols, RandomSymbol,
from sympy.stats.stochastic_process import StochasticPSpace
from sympy.stats.symbolic_probability import Probability, Expectation
from sympy.stats.frv_types import Bernoulli, BernoulliDistribution, FiniteRV
from sympy.stats.drv_types import Poisson, PoissonDistribution
from sympy.stats.crv_types import Normal, NormalDistribution, Gamma, GammaDistribution
from sympy.core.sympify import _sympify, sympify
@classmethod
def _sanity_checks(cls, state_space, trans_probs):
    if state_space is None and trans_probs is None:
        _n = Dummy('n', integer=True, nonnegative=True)
        state_space = _state_converter(Range(_n))
        trans_probs = _matrix_checks(MatrixSymbol('_T', _n, _n))
    elif state_space is None:
        trans_probs = _matrix_checks(trans_probs)
        state_space = _state_converter(Range(trans_probs.shape[0]))
    elif trans_probs is None:
        state_space = _state_converter(state_space)
        if isinstance(state_space, Range):
            _n = ceiling((state_space.stop - state_space.start) / state_space.step)
        else:
            _n = len(state_space)
        trans_probs = MatrixSymbol('_T', _n, _n)
    else:
        state_space = _state_converter(state_space)
        trans_probs = _matrix_checks(trans_probs)
        if isinstance(state_space, Range):
            ss_size = ceiling((state_space.stop - state_space.start) / state_space.step)
        else:
            ss_size = len(state_space)
        if ss_size != trans_probs.shape[0]:
            raise ValueError('The size of the state space and the number of rows of the transition matrix must be the same.')
    return (state_space, trans_probs)