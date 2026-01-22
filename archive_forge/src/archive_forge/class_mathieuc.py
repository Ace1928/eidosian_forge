from sympy.core.function import Function, ArgumentIndexError
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos
class mathieuc(MathieuBase):
    """
    The Mathieu Cosine function $C(a,q,z)$.

    Explanation
    ===========

    This function is one solution of the Mathieu differential equation:

    .. math ::
        y(x)^{\\prime\\prime} + (a - 2 q \\cos(2 x)) y(x) = 0

    The other solution is the Mathieu Sine function.

    Examples
    ========

    >>> from sympy import diff, mathieuc
    >>> from sympy.abc import a, q, z

    >>> mathieuc(a, q, z)
    mathieuc(a, q, z)

    >>> mathieuc(a, 0, z)
    cos(sqrt(a)*z)

    >>> diff(mathieuc(a, q, z), z)
    mathieucprime(a, q, z)

    See Also
    ========

    mathieus: Mathieu sine function
    mathieusprime: Derivative of Mathieu sine function
    mathieucprime: Derivative of Mathieu cosine function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathieu_function
    .. [2] https://dlmf.nist.gov/28
    .. [3] https://mathworld.wolfram.com/MathieuFunction.html
    .. [4] https://functions.wolfram.com/MathieuandSpheroidalFunctions/MathieuC/

    """

    def fdiff(self, argindex=1):
        if argindex == 3:
            a, q, z = self.args
            return mathieucprime(a, q, z)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, a, q, z):
        if q.is_Number and q.is_zero:
            return cos(sqrt(a) * z)
        if z.could_extract_minus_sign():
            return cls(a, q, -z)