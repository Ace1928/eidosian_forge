from sympy.core import Add, Mul, S
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import I
from sympy.core.relational import Eq, Equality
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.function import (expand_mul, expand, Derivative,
from sympy.functions import (exp, im, cos, sin, re, Piecewise,
from sympy.functions.combinatorial.factorials import factorial
from sympy.matrices import zeros, Matrix, NonSquareMatrixError, MatrixBase, eye
from sympy.polys import Poly, together
from sympy.simplify import collect, radsimp, signsimp # type: ignore
from sympy.simplify.powsimp import powdenest, powsimp
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify
from sympy.sets.sets import FiniteSet
from sympy.solvers.deutils import ode_order
from sympy.solvers.solveset import NonlinearError, solveset
from sympy.utilities.iterables import (connected_components, iterable,
from sympy.utilities.misc import filldedent
from sympy.integrals.integrals import Integral, integrate
def _higher_order_to_first_order(eqs, sys_order, t, funcs=None, type='type0', **kwargs):
    if funcs is None:
        funcs = sys_order.keys()
    if type == 'type1':
        t_ = Symbol('{}_'.format(t))
        new_funcs = [Function(Dummy('{}_'.format(f.func.__name__)))(t_) for f in funcs]
        max_order = max((sys_order[func] for func in funcs))
        subs_dict = {func: new_func for func, new_func in zip(funcs, new_funcs)}
        subs_dict[t] = exp(t_)
        free_function = Function(Dummy())

        def _get_coeffs_from_subs_expression(expr):
            if isinstance(expr, Subs):
                free_symbol = expr.args[1][0]
                term = expr.args[0]
                return {ode_order(term, free_symbol): 1}
            if isinstance(expr, Mul):
                coeff = expr.args[0]
                order = list(_get_coeffs_from_subs_expression(expr.args[1]).keys())[0]
                return {order: coeff}
            if isinstance(expr, Add):
                coeffs = {}
                for arg in expr.args:
                    if isinstance(arg, Mul):
                        coeffs.update(_get_coeffs_from_subs_expression(arg))
                    else:
                        order = list(_get_coeffs_from_subs_expression(arg).keys())[0]
                        coeffs[order] = 1
                return coeffs
        for o in range(1, max_order + 1):
            expr = free_function(log(t_)).diff(t_, o) * t_ ** o
            coeff_dict = _get_coeffs_from_subs_expression(expr)
            coeffs = [coeff_dict[order] if order in coeff_dict else 0 for order in range(o + 1)]
            expr_to_subs = sum((free_function(t_).diff(t_, i) * c for i, c in enumerate(coeffs))) / t ** o
            subs_dict.update({f.diff(t, o): expr_to_subs.subs(free_function(t_), nf) for f, nf in zip(funcs, new_funcs)})
        new_eqs = [eq.subs(subs_dict) for eq in eqs]
        new_sys_order = {nf: sys_order[f] for f, nf in zip(funcs, new_funcs)}
        new_eqs = canonical_odes(new_eqs, new_funcs, t_)[0]
        return _higher_order_to_first_order(new_eqs, new_sys_order, t_, funcs=new_funcs)
    if type == 'type2':
        J = kwargs.get('J', None)
        f_t = kwargs.get('f_t', None)
        b = kwargs.get('b', None)
        P = kwargs.get('P', None)
        max_order = max((sys_order[func] for func in funcs))
        return _higher_order_type2_to_sub_systems(J, f_t, funcs, t, max_order, P=P, b=b)
    new_funcs = []
    for prev_func in funcs:
        func_name = prev_func.func.__name__
        func = Function(Dummy('{}_0'.format(func_name)))(t)
        new_funcs.append(func)
        subs_dict = {prev_func: func}
        new_eqs = []
        for i in range(1, sys_order[prev_func]):
            new_func = Function(Dummy('{}_{}'.format(func_name, i)))(t)
            subs_dict[prev_func.diff(t, i)] = new_func
            new_funcs.append(new_func)
            prev_f = subs_dict[prev_func.diff(t, i - 1)]
            new_eq = Eq(prev_f.diff(t), new_func)
            new_eqs.append(new_eq)
        eqs = [eq.subs(subs_dict) for eq in eqs] + new_eqs
    return (eqs, new_funcs)