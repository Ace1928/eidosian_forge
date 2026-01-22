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
class ContinuousMarkovChain(ContinuousTimeStochasticProcess, MarkovProcess):
    """
    Represents continuous time Markov chain.

    Parameters
    ==========

    sym : Symbol/str
    state_space : Set
        Optional, by default, S.Reals
    gen_mat : Matrix/ImmutableMatrix/MatrixSymbol
        Optional, by default, None

    Examples
    ========

    >>> from sympy.stats import ContinuousMarkovChain, P
    >>> from sympy import Matrix, S, Eq, Gt
    >>> G = Matrix([[-S(1), S(1)], [S(1), -S(1)]])
    >>> C = ContinuousMarkovChain('C', state_space=[0, 1], gen_mat=G)
    >>> C.limiting_distribution()
    Matrix([[1/2, 1/2]])
    >>> C.state_space
    {0, 1}
    >>> C.generator_matrix
    Matrix([
    [-1,  1],
    [ 1, -1]])

    Probability queries are supported

    >>> P(Eq(C(1.96), 0), Eq(C(0.78), 1)).round(5)
    0.45279
    >>> P(Gt(C(1.7), 0), Eq(C(0.82), 1)).round(5)
    0.58602

    Probability of expressions with multiple RandomIndexedSymbols
    can also be calculated provided there is only 1 RandomIndexedSymbol
    in the given condition. It is always better to use Rational instead
    of floating point numbers for the probabilities in the
    generator matrix to avoid errors.

    >>> from sympy import Gt, Le, Rational
    >>> G = Matrix([[-S(1), Rational(1, 10), Rational(9, 10)], [Rational(2, 5), -S(1), Rational(3, 5)], [Rational(1, 2), Rational(1, 2), -S(1)]])
    >>> C = ContinuousMarkovChain('C', state_space=[0, 1, 2], gen_mat=G)
    >>> P(Eq(C(3.92), C(1.75)), Eq(C(0.46), 0)).round(5)
    0.37933
    >>> P(Gt(C(3.92), C(1.75)), Eq(C(0.46), 0)).round(5)
    0.34211
    >>> P(Le(C(1.57), C(3.14)), Eq(C(1.22), 1)).round(4)
    0.7143

    Symbolic probability queries are also supported

    >>> from sympy import symbols
    >>> a,b,c,d = symbols('a b c d')
    >>> G = Matrix([[-S(1), Rational(1, 10), Rational(9, 10)], [Rational(2, 5), -S(1), Rational(3, 5)], [Rational(1, 2), Rational(1, 2), -S(1)]])
    >>> C = ContinuousMarkovChain('C', state_space=[0, 1, 2], gen_mat=G)
    >>> query = P(Eq(C(a), b), Eq(C(c), d))
    >>> query.subs({a:3.65, b:2, c:1.78, d:1}).evalf().round(10)
    0.4002723175
    >>> P(Eq(C(3.65), 2), Eq(C(1.78), 1)).round(10)
    0.4002723175
    >>> query_gt = P(Gt(C(a), b), Eq(C(c), d))
    >>> query_gt.subs({a:43.2, b:0, c:3.29, d:2}).evalf().round(10)
    0.6832579186
    >>> P(Gt(C(43.2), 0), Eq(C(3.29), 2)).round(10)
    0.6832579186

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Markov_chain#Continuous-time_Markov_chain
    .. [2] https://u.math.biu.ac.il/~amirgi/CTMCnotes.pdf
    """
    index_set = S.Reals

    def __new__(cls, sym, state_space=None, gen_mat=None):
        sym = _symbol_converter(sym)
        state_space, gen_mat = MarkovProcess._sanity_checks(state_space, gen_mat)
        obj = Basic.__new__(cls, sym, state_space, gen_mat)
        indices = {}
        if isinstance(obj.number_of_states, Integer):
            for index, state in enumerate(obj.state_space):
                indices[state] = index
        obj.index_of = indices
        return obj

    @property
    def generator_matrix(self):
        return self.args[2]

    @cacheit
    def transition_probabilities(self, gen_mat=None):
        t = Dummy('t')
        if isinstance(gen_mat, (Matrix, ImmutableMatrix)) and gen_mat.is_diagonalizable():
            Q, D = gen_mat.diagonalize()
            return Lambda(t, Q * exp(t * D) * Q.inv())
        if gen_mat != None:
            return Lambda(t, exp(t * gen_mat))

    def limiting_distribution(self):
        gen_mat = self.generator_matrix
        if gen_mat is None:
            return None
        if isinstance(gen_mat, MatrixSymbol):
            wm = MatrixSymbol('wm', 1, gen_mat.shape[0])
            return Lambda((wm, gen_mat), Eq(wm * gen_mat, wm))
        w = IndexedBase('w')
        wi = [w[i] for i in range(gen_mat.shape[0])]
        wm = Matrix([wi])
        eqs = (wm * gen_mat).tolist()[0]
        eqs.append(sum(wi) - 1)
        soln = list(linsolve(eqs, wi))[0]
        return ImmutableMatrix([soln])