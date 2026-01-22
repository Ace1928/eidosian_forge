from sympy.core import Add, Expr, Mul, S, sympify
from sympy.core.function import _mexpand, count_ops, expand_mul
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.functions import root, sign, sqrt
from sympy.polys import Poly, PolynomialError
def _denester(nested, av0, h, max_depth_level):
    """Denests a list of expressions that contain nested square roots.

    Explanation
    ===========

    Algorithm based on <http://www.almaden.ibm.com/cs/people/fagin/symb85.pdf>.

    It is assumed that all of the elements of 'nested' share the same
    bottom-level radicand. (This is stated in the paper, on page 177, in
    the paragraph immediately preceding the algorithm.)

    When evaluating all of the arguments in parallel, the bottom-level
    radicand only needs to be denested once. This means that calling
    _denester with x arguments results in a recursive invocation with x+1
    arguments; hence _denester has polynomial complexity.

    However, if the arguments were evaluated separately, each call would
    result in two recursive invocations, and the algorithm would have
    exponential complexity.

    This is discussed in the paper in the middle paragraph of page 179.
    """
    from sympy.simplify.simplify import radsimp
    if h > max_depth_level:
        return (None, None)
    if av0[1] is None:
        return (None, None)
    if av0[0] is None and all((n.is_Number for n in nested)):
        for f in _subsets(len(nested)):
            p = _mexpand(Mul(*[nested[i] for i in range(len(f)) if f[i]]))
            if f.count(1) > 1 and f[-1]:
                p = -p
            sqp = sqrt(p)
            if sqp.is_Rational:
                return (sqp, f)
        return (sqrt(nested[-1]), [0] * len(nested))
    else:
        R = None
        if av0[0] is not None:
            values = [av0[:2]]
            R = av0[2]
            nested2 = [av0[3], R]
            av0[0] = None
        else:
            values = list(filter(None, [_sqrt_match(expr) for expr in nested]))
            for v in values:
                if v[2]:
                    if R is not None:
                        if R != v[2]:
                            av0[1] = None
                            return (None, None)
                    else:
                        R = v[2]
            if R is None:
                return (sqrt(nested[-1]), [0] * len(nested))
            nested2 = [_mexpand(v[0] ** 2) - _mexpand(R * v[1] ** 2) for v in values] + [R]
        d, f = _denester(nested2, av0, h + 1, max_depth_level)
        if not f:
            return (None, None)
        if not any((f[i] for i in range(len(nested)))):
            v = values[-1]
            return (sqrt(v[0] + _mexpand(v[1] * d)), f)
        else:
            p = Mul(*[nested[i] for i in range(len(nested)) if f[i]])
            v = _sqrt_match(p)
            if 1 in f and f.index(1) < len(nested) - 1 and f[len(nested) - 1]:
                v[0] = -v[0]
                v[1] = -v[1]
            if not f[len(nested)]:
                vad = _mexpand(v[0] + d)
                if vad <= 0:
                    return (sqrt(nested[-1]), [0] * len(nested))
                if not (sqrt_depth(vad) <= sqrt_depth(R) + 1 or (vad ** 2).is_Number):
                    av0[1] = None
                    return (None, None)
                sqvad = _sqrtdenest1(sqrt(vad), denester=False)
                if not sqrt_depth(sqvad) <= sqrt_depth(R) + 1:
                    av0[1] = None
                    return (None, None)
                sqvad1 = radsimp(1 / sqvad)
                res = _mexpand(sqvad / sqrt(2) + v[1] * sqrt(R) * sqvad1 / sqrt(2))
                return (res, f)
            else:
                s2 = _mexpand(v[1] * R) + d
                if s2 <= 0:
                    return (sqrt(nested[-1]), [0] * len(nested))
                FR, s = (root(_mexpand(R), 4), sqrt(s2))
                return (_mexpand(s / (sqrt(2) * FR) + v[0] * FR / (sqrt(2) * s)), f)