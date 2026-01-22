from sympy.core.function import Function, ArgumentIndexError
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos
class mathieucprime(MathieuBase):
    """
    The derivative $C^{\\prime}(a,q,z)$ of the Mathieu Cosine function.

    Explanation
    ===========

    This function is one solution of the Mathieu differential equation:

    .. math ::
        y(x)^{\\prime\\prime} + (a - 2 q \\cos(2 x)) y(x) = 0

    The other solution is the Mathieu Sine function.

    Examples
    ========

    >>> from sympy import diff, mathieucprime
    >>> from sympy.abc import a, q, z

    >>> mathieucprime(a, q, z)
    mathieucprime(a, q, z)

    >>> mathieucprime(a, 0, z)
    -sqrt(a)*sin(sqrt(a)*z)

    >>> diff(mathieucprime(a, q, z), z)
    (-a + 2*q*cos(2*z))*mathieuc(a, q, z)

    See Also
    ========

    mathieus: Mathieu sine function
    mathieuc: Mathieu cosine function
    mathieusprime: Derivative of Mathieu sine function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathieu_function
    .. [2] https://dlmf.nist.gov/28
    .. [3] https://mathworld.wolfram.com/MathieuFunction.html
    .. [4] https://functions.wolfram.com/MathieuandSpheroidalFunctions/MathieuCPrime/

    """

    def fdiff(self, argindex=1):
        if argindex == 3:
            a, q, z = self.args
            return (2 * q * cos(2 * z) - a) * mathieuc(a, q, z)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, a, q, z):
        if q.is_Number and q.is_zero:
            return -sqrt(a) * sin(sqrt(a) * z)
        if z.could_extract_minus_sign():
            return -cls(a, q, -z)