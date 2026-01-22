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
class CountingProcess(ContinuousTimeStochasticProcess):
    """
    This class handles the common methods of the Counting Processes
    such as Poisson, Wiener and Gamma Processes
    """
    index_set = _set_converter(Interval(0, oo))

    @property
    def symbol(self):
        return self.args[0]

    def expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Computes expectation

        Parameters
        ==========

        expr: RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition: Relational, Boolean
            The given conditions under which computations should be done, i.e,
            the intervals on which each variable time stamp in expr is defined

        Returns
        =======

        Expectation of the given expr

        """
        if condition is not None:
            intervals, rv_swap = get_timerv_swaps(expr, condition)
            if len(intervals) == 1 or all((Intersection(*intv_comb) == EmptySet for intv_comb in itertools.combinations(intervals, 2))):
                if expr.is_Add:
                    return Add.fromiter((self.expectation(arg, condition) for arg in expr.args))
                expr = expr.subs(rv_swap)
            else:
                return Expectation(expr, condition)
        return _SubstituteRV._expectation(expr, evaluate=evaluate, **kwargs)

    def _solve_argwith_tworvs(self, arg):
        if arg.args[0].key >= arg.args[1].key or isinstance(arg, Eq):
            diff_key = abs(arg.args[0].key - arg.args[1].key)
            rv = arg.args[0]
            arg = arg.__class__(rv.pspace.process(diff_key), 0)
        else:
            diff_key = arg.args[1].key - arg.args[0].key
            rv = arg.args[1]
            arg = arg.__class__(rv.pspace.process(diff_key), 0)
        return arg

    def _solve_numerical(self, condition, given_condition=None):
        if isinstance(condition, And):
            args_list = list(condition.args)
        else:
            args_list = [condition]
        if given_condition is not None:
            if isinstance(given_condition, And):
                args_list.extend(list(given_condition.args))
            else:
                args_list.extend([given_condition])
        args_list = sorted(args_list, key=lambda x: x.args[0].key)
        result = []
        cond_args = list(condition.args) if isinstance(condition, And) else [condition]
        if args_list[0] in cond_args and (not (is_random(args_list[0].args[0]) and is_random(args_list[0].args[1]))):
            result.append(_SubstituteRV._probability(args_list[0]))
        if is_random(args_list[0].args[0]) and is_random(args_list[0].args[1]):
            arg = self._solve_argwith_tworvs(args_list[0])
            result.append(_SubstituteRV._probability(arg))
        for i in range(len(args_list) - 1):
            curr, nex = (args_list[i], args_list[i + 1])
            diff_key = nex.args[0].key - curr.args[0].key
            working_set = curr.args[0].pspace.process.state_space
            if curr.args[1] > nex.args[1]:
                result.append(0)
                break
            if isinstance(curr, Eq):
                working_set = Intersection(working_set, Interval.Lopen(curr.args[1], oo))
            else:
                working_set = Intersection(working_set, curr.as_set())
            if isinstance(nex, Eq):
                working_set = Intersection(working_set, Interval(-oo, nex.args[1]))
            else:
                working_set = Intersection(working_set, nex.as_set())
            if working_set == EmptySet:
                rv = Eq(curr.args[0].pspace.process(diff_key), 0)
                result.append(_SubstituteRV._probability(rv))
            elif working_set.is_finite_set:
                if isinstance(curr, Eq) and isinstance(nex, Eq):
                    rv = Eq(curr.args[0].pspace.process(diff_key), len(working_set))
                    result.append(_SubstituteRV._probability(rv))
                elif isinstance(curr, Eq) ^ isinstance(nex, Eq):
                    result.append(Add.fromiter((_SubstituteRV._probability(Eq(curr.args[0].pspace.process(diff_key), x)) for x in range(len(working_set)))))
                else:
                    n = len(working_set)
                    result.append(Add.fromiter(((n - x) * _SubstituteRV._probability(Eq(curr.args[0].pspace.process(diff_key), x)) for x in range(n))))
            else:
                result.append(_SubstituteRV._probability(curr.args[0].pspace.process(diff_key) <= working_set._sup - working_set._inf))
        return Mul.fromiter(result)

    def probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        Computes probability.

        Parameters
        ==========

        condition: Relational
            Condition for which probability has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        given_condition: Relational, Boolean
            The given conditions under which computations should be done, i.e,
            the intervals on which each variable time stamp in expr is defined

        Returns
        =======

        Probability of the condition

        """
        check_numeric = True
        if isinstance(condition, (And, Or)):
            cond_args = condition.args
        else:
            cond_args = (condition,)
        if not all((arg.args[0].key.is_number for arg in cond_args)):
            check_numeric = False
        if given_condition is not None:
            check_given_numeric = True
            if isinstance(given_condition, (And, Or)):
                given_cond_args = given_condition.args
            else:
                given_cond_args = (given_condition,)
            if given_condition.has(Contains):
                check_given_numeric = False
            if check_numeric and check_given_numeric:
                res = []
                if isinstance(condition, Or):
                    res.append(Add.fromiter((self._solve_numerical(arg, given_condition) for arg in condition.args)))
                if isinstance(given_condition, Or):
                    res.append(Add.fromiter((self._solve_numerical(condition, arg) for arg in given_condition.args)))
                if res:
                    return Add.fromiter(res)
                return self._solve_numerical(condition, given_condition)
            if not all((arg.has(Contains) for arg in given_cond_args)):
                raise ValueError('If given condition is passed with `Contains`, then please pass the evaluated condition with its corresponding information in terms of intervals of each time stamp to be passed in given condition.')
            intervals, rv_swap = get_timerv_swaps(condition, given_condition)
            if len(intervals) == 1 or all((Intersection(*intv_comb) == EmptySet for intv_comb in itertools.combinations(intervals, 2))):
                if isinstance(condition, And):
                    return Mul.fromiter((self.probability(arg, given_condition) for arg in condition.args))
                elif isinstance(condition, Or):
                    return Add.fromiter((self.probability(arg, given_condition) for arg in condition.args))
                condition = condition.subs(rv_swap)
            else:
                return Probability(condition, given_condition)
        if check_numeric:
            return self._solve_numerical(condition)
        return _SubstituteRV._probability(condition, evaluate=evaluate, **kwargs)