from itertools import combinations, product, zip_longest
from sympy.assumptions.assume import AppliedPredicate, Predicate
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.core.singleton import S
from sympy.logic.boolalg import Or, And, Not, Xnor
from sympy.logic.boolalg import (Equivalent, ITE, Implies, Nand, Nor, Xor)
def to_NNF(expr, composite_map=None):
    """
    Generates the Negation Normal Form of any boolean expression in terms
    of AND, OR, and Literal objects.

    Examples
    ========

    >>> from sympy import Q, Eq
    >>> from sympy.assumptions.cnf import to_NNF
    >>> from sympy.abc import x, y
    >>> expr = Q.even(x) & ~Q.positive(x)
    >>> to_NNF(expr)
    (Literal(Q.even(x), False) & Literal(Q.positive(x), True))

    Supported boolean objects are converted to corresponding predicates.

    >>> to_NNF(Eq(x, y))
    Literal(Q.eq(x, y), False)

    If ``composite_map`` argument is given, ``to_NNF`` decomposes the
    specified predicate into a combination of primitive predicates.

    >>> cmap = {Q.nonpositive: Q.negative | Q.zero}
    >>> to_NNF(Q.nonpositive, cmap)
    (Literal(Q.negative, False) | Literal(Q.zero, False))
    >>> to_NNF(Q.nonpositive(x), cmap)
    (Literal(Q.negative(x), False) | Literal(Q.zero(x), False))
    """
    from sympy.assumptions.ask import Q
    if composite_map is None:
        composite_map = {}
    binrelpreds = {Eq: Q.eq, Ne: Q.ne, Gt: Q.gt, Lt: Q.lt, Ge: Q.ge, Le: Q.le}
    if type(expr) in binrelpreds:
        pred = binrelpreds[type(expr)]
        expr = pred(*expr.args)
    if isinstance(expr, Not):
        arg = expr.args[0]
        tmp = to_NNF(arg, composite_map)
        return ~tmp
    if isinstance(expr, Or):
        return OR(*[to_NNF(x, composite_map) for x in Or.make_args(expr)])
    if isinstance(expr, And):
        return AND(*[to_NNF(x, composite_map) for x in And.make_args(expr)])
    if isinstance(expr, Nand):
        tmp = AND(*[to_NNF(x, composite_map) for x in expr.args])
        return ~tmp
    if isinstance(expr, Nor):
        tmp = OR(*[to_NNF(x, composite_map) for x in expr.args])
        return ~tmp
    if isinstance(expr, Xor):
        cnfs = []
        for i in range(0, len(expr.args) + 1, 2):
            for neg in combinations(expr.args, i):
                clause = [~to_NNF(s, composite_map) if s in neg else to_NNF(s, composite_map) for s in expr.args]
                cnfs.append(OR(*clause))
        return AND(*cnfs)
    if isinstance(expr, Xnor):
        cnfs = []
        for i in range(0, len(expr.args) + 1, 2):
            for neg in combinations(expr.args, i):
                clause = [~to_NNF(s, composite_map) if s in neg else to_NNF(s, composite_map) for s in expr.args]
                cnfs.append(OR(*clause))
        return ~AND(*cnfs)
    if isinstance(expr, Implies):
        L, R = (to_NNF(expr.args[0], composite_map), to_NNF(expr.args[1], composite_map))
        return OR(~L, R)
    if isinstance(expr, Equivalent):
        cnfs = []
        for a, b in zip_longest(expr.args, expr.args[1:], fillvalue=expr.args[0]):
            a = to_NNF(a, composite_map)
            b = to_NNF(b, composite_map)
            cnfs.append(OR(~a, b))
        return AND(*cnfs)
    if isinstance(expr, ITE):
        L = to_NNF(expr.args[0], composite_map)
        M = to_NNF(expr.args[1], composite_map)
        R = to_NNF(expr.args[2], composite_map)
        return AND(OR(~L, M), OR(L, R))
    if isinstance(expr, AppliedPredicate):
        pred, args = (expr.function, expr.arguments)
        newpred = composite_map.get(pred, None)
        if newpred is not None:
            return to_NNF(newpred.rcall(*args), composite_map)
    if isinstance(expr, Predicate):
        newpred = composite_map.get(expr, None)
        if newpred is not None:
            return to_NNF(newpred, composite_map)
    return Literal(expr)