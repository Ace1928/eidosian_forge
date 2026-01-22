from sympy.core.function import Derivative
from sympy.core.singleton import S
from sympy.core.function import Subs
from sympy.core.traversal import preorder_traversal
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable
def _as_finite_diff(derivative, points=1, x0=None, wrt=None):
    """
    Returns an approximation of a derivative of a function in
    the form of a finite difference formula. The expression is a
    weighted sum of the function at a number of discrete values of
    (one of) the independent variable(s).

    Parameters
    ==========

    derivative: a Derivative instance

    points: sequence or coefficient, optional
        If sequence: discrete values (length >= order+1) of the
        independent variable used for generating the finite
        difference weights.
        If it is a coefficient, it will be used as the step-size
        for generating an equidistant sequence of length order+1
        centered around ``x0``. default: 1 (step-size 1)

    x0: number or Symbol, optional
        the value of the independent variable (``wrt``) at which the
        derivative is to be approximated. Default: same as ``wrt``.

    wrt: Symbol, optional
        "with respect to" the variable for which the (partial)
        derivative is to be approximated for. If not provided it
        is required that the Derivative is ordinary. Default: ``None``.

    Examples
    ========

    >>> from sympy import symbols, Function, exp, sqrt, Symbol
    >>> from sympy.calculus.finite_diff import _as_finite_diff
    >>> x, h = symbols('x h')
    >>> f = Function('f')
    >>> _as_finite_diff(f(x).diff(x))
    -f(x - 1/2) + f(x + 1/2)

    The default step size and number of points are 1 and ``order + 1``
    respectively. We can change the step size by passing a symbol
    as a parameter:

    >>> _as_finite_diff(f(x).diff(x), h)
    -f(-h/2 + x)/h + f(h/2 + x)/h

    We can also specify the discretized values to be used in a sequence:

    >>> _as_finite_diff(f(x).diff(x), [x, x+h, x+2*h])
    -3*f(x)/(2*h) + 2*f(h + x)/h - f(2*h + x)/(2*h)

    The algorithm is not restricted to use equidistant spacing, nor
    do we need to make the approximation around ``x0``, but we can get
    an expression estimating the derivative at an offset:

    >>> e, sq2 = exp(1), sqrt(2)
    >>> xl = [x-h, x+h, x+e*h]
    >>> _as_finite_diff(f(x).diff(x, 1), xl, x+h*sq2)
    2*h*((h + sqrt(2)*h)/(2*h) - (-sqrt(2)*h + h)/(2*h))*f(E*h + x)/((-h + E*h)*(h + E*h)) +
    (-(-sqrt(2)*h + h)/(2*h) - (-sqrt(2)*h + E*h)/(2*h))*f(-h + x)/(h + E*h) +
    (-(h + sqrt(2)*h)/(2*h) + (-sqrt(2)*h + E*h)/(2*h))*f(h + x)/(-h + E*h)

    Partial derivatives are also supported:

    >>> y = Symbol('y')
    >>> d2fdxdy=f(x,y).diff(x,y)
    >>> _as_finite_diff(d2fdxdy, wrt=x)
    -Derivative(f(x - 1/2, y), y) + Derivative(f(x + 1/2, y), y)

    See also
    ========

    sympy.calculus.finite_diff.apply_finite_diff
    sympy.calculus.finite_diff.finite_diff_weights

    """
    if derivative.is_Derivative:
        pass
    elif derivative.is_Atom:
        return derivative
    else:
        return derivative.fromiter([_as_finite_diff(ar, points, x0, wrt) for ar in derivative.args], **derivative.assumptions0)
    if wrt is None:
        old = None
        for v in derivative.variables:
            if old is v:
                continue
            derivative = _as_finite_diff(derivative, points, x0, v)
            old = v
        return derivative
    order = derivative.variables.count(wrt)
    if x0 is None:
        x0 = wrt
    if not iterable(points):
        if getattr(points, 'is_Function', False) and wrt in points.args:
            points = points.subs(wrt, x0)
        if order % 2 == 0:
            points = [x0 + points * i for i in range(-order // 2, order // 2 + 1)]
        else:
            points = [x0 + points * S(i) / 2 for i in range(-order, order + 1, 2)]
    others = [wrt, 0]
    for v in set(derivative.variables):
        if v == wrt:
            continue
        others += [v, derivative.variables.count(v)]
    if len(points) < order + 1:
        raise ValueError('Too few points for order %d' % order)
    return apply_finite_diff(order, points, [Derivative(derivative.expr.subs({wrt: x}), *others) for x in points], x0)