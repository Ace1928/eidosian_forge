from .z3 import *

    Return the non-constant subresultants of 'p' and 'q' with respect to the "variable" 'x'.

    'p', 'q' and 'x' are Z3 expressions where 'p' and 'q' are arithmetic terms.
    Note that, any subterm that cannot be viewed as a polynomial is assumed to be a variable.
    Example: f(a) is a considered to be a variable b in the polynomial

    f(a)*f(a) + 2*f(a) + 1

    >>> x, y = Reals('x y')
    >>> subresultants(2*x + y, 3*x - 2*y + 2, x)
    [-7*y + 4]
    >>> r = subresultants(3*y*x**2 + y**3 + 1, 2*x**3 + y + 3, x)
    >>> r[0]
    4*y**9 + 12*y**6 + 27*y**5 + 162*y**4 + 255*y**3 + 4
    >>> r[1]
    -6*y**4 + -6*y
    