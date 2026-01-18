from sympy.core import S, sympify
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int

    Compute the coefficients of n'th term in linear recursion
    sequence defined by c.

    `x^k = c_0 x^{k-1} + c_1 x^{k-2} + \cdots + c_{k-1}`.

    It computes the coefficients by using binary exponentiation.
    This function is used by `linrec` and `_eval_pow_by_cayley`.

    Parameters
    ==========

    c = coefficients of the divisor polynomial
    n = exponent of x, so dividend is x^n

    