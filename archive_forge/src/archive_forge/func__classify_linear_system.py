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
def _classify_linear_system(eqs, funcs, t, is_canon=False):
    """
    Returns a dictionary with details of the eqs if the system passed is linear
    and can be classified by this function else returns None

    Explanation
    ===========

    This function takes the eqs, converts it into a form Ax = b where x is a vector of terms
    containing dependent variables and their derivatives till their maximum order. If it is
    possible to convert eqs into Ax = b, then all the equations in eqs are linear otherwise
    they are non-linear.

    To check if the equations are constant coefficient, we need to check if all the terms in
    A obtained above are constant or not.

    To check if the equations are homogeneous or not, we need to check if b is a zero matrix
    or not.

    Parameters
    ==========

    eqs: List
        List of ODEs
    funcs: List
        List of dependent variables
    t: Symbol
        Independent variable of the equations in eqs
    is_canon: Boolean
        If True, then this function will not try to get the
        system in canonical form. Default value is False

    Returns
    =======

    match = {
        'no_of_equation': len(eqs),
        'eq': eqs,
        'func': funcs,
        'order': order,
        'is_linear': is_linear,
        'is_constant': is_constant,
        'is_homogeneous': is_homogeneous,
    }

    Dict or list of Dicts or None
        Dict with values for keys:
            1. no_of_equation: Number of equations
            2. eq: The set of equations
            3. func: List of dependent variables
            4. order: A dictionary that gives the order of the
                      dependent variable in eqs
            5. is_linear: Boolean value indicating if the set of
                          equations are linear or not.
            6. is_constant: Boolean value indicating if the set of
                          equations have constant coefficients or not.
            7. is_homogeneous: Boolean value indicating if the set of
                          equations are homogeneous or not.
            8. commutative_antiderivative: Antiderivative of the coefficient
                          matrix if the coefficient matrix is non-constant
                          and commutative with its antiderivative. This key
                          may or may not exist.
            9. is_general: Boolean value indicating if the system of ODEs is
                           solvable using one of the general case solvers or not.
            10. rhs: rhs of the non-homogeneous system of ODEs in Matrix form. This
                     key may or may not exist.
            11. is_higher_order: True if the system passed has an order greater than 1.
                                 This key may or may not exist.
            12. is_second_order: True if the system passed is a second order ODE. This
                                 key may or may not exist.
        This Dict is the answer returned if the eqs are linear and constant
        coefficient. Otherwise, None is returned.

    """
    if len(funcs) != len(eqs):
        raise ValueError('Number of functions given is not equal to the number of equations %s' % funcs)
    for func in funcs:
        if len(func.args) != 1:
            raise ValueError('dsolve() and classify_sysode() work with functions of one variable only, not %s' % func)
    order = _get_func_order(eqs, funcs)
    system_order = max((order[func] for func in funcs))
    is_higher_order = system_order > 1
    is_second_order = system_order == 2 and all((order[func] == 2 for func in funcs))
    try:
        canon_eqs = canonical_odes(eqs, funcs, t) if not is_canon else [eqs]
        if len(canon_eqs) == 1:
            As, b = linear_ode_to_matrix(canon_eqs[0], funcs, t, system_order)
        else:
            match = {'is_implicit': True, 'canon_eqs': canon_eqs}
            return match
    except ODENonlinearError:
        return None
    is_linear = True
    is_homogeneous = True if b.is_zero_matrix else False
    match = {'no_of_equation': len(eqs), 'eq': eqs, 'func': funcs, 'order': order, 'is_linear': is_linear, 'is_homogeneous': is_homogeneous, 'is_general': True}
    if not is_homogeneous:
        match['rhs'] = b
    is_constant = all((_matrix_is_constant(A_, t) for A_ in As))
    if not is_higher_order:
        A = As[1]
        match['func_coeff'] = A
        is_constant = _matrix_is_constant(A, t)
        match['is_constant'] = is_constant
        try:
            system_info = linodesolve_type(A, t, b=b)
        except NotImplementedError:
            return None
        match.update(system_info)
        antiderivative = match.pop('antiderivative')
        if not is_constant:
            match['commutative_antiderivative'] = antiderivative
        return match
    else:
        match['type_of_equation'] = 'type0'
        if is_second_order:
            A1, A0 = As[1:]
            match_second_order = _match_second_order_type(A1, A0, t)
            match.update(match_second_order)
            match['is_second_order'] = True
        if match['type_of_equation'] == 'type0' and (not is_constant):
            is_euler = _is_euler_system(As, t)
            if is_euler:
                t_ = Symbol('{}_'.format(t))
                match.update({'is_transformed': True, 'type_of_equation': 'type1', 't_': t_})
            else:
                is_jordan = lambda M: M == Matrix.jordan_block(M.shape[0], M[0, 0])
                terms = _factor_matrix(As[-1], t)
                if all((A.is_zero_matrix for A in As[1:-1])) and terms is not None and (not is_jordan(terms[1])):
                    P, J = terms[1].jordan_form()
                    match.update({'type_of_equation': 'type2', 'J': J, 'f(t)': terms[0], 'P': P, 'is_transformed': True})
            if match['type_of_equation'] != 'type0' and is_second_order:
                match.pop('is_second_order', None)
        match['is_higher_order'] = is_higher_order
        return match