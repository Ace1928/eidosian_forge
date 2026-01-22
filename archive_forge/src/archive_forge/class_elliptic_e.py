from sympy.core import S, pi, I, Rational
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.hyperbolic import atanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, tan
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg
class elliptic_e(Function):
    """
    Called with two arguments $z$ and $m$, evaluates the
    incomplete elliptic integral of the second kind, defined by

    .. math:: E\\left(z\\middle| m\\right) = \\int_0^z \\sqrt{1 - m \\sin^2 t} dt

    Called with a single argument $m$, evaluates the Legendre complete
    elliptic integral of the second kind

    .. math:: E(m) = E\\left(\\tfrac{\\pi}{2}\\middle| m\\right)

    Explanation
    ===========

    The function $E(m)$ is a single-valued function on the complex
    plane with branch cut along the interval $(1, \\infty)$.

    Note that our notation defines the incomplete elliptic integral
    in terms of the parameter $m$ instead of the elliptic modulus
    (eccentricity) $k$.
    In this case, the parameter $m$ is defined as $m=k^2$.

    Examples
    ========

    >>> from sympy import elliptic_e, I
    >>> from sympy.abc import z, m
    >>> elliptic_e(z, m).series(z)
    z + z**5*(-m**2/40 + m/30) - m*z**3/6 + O(z**6)
    >>> elliptic_e(m).series(n=4)
    pi/2 - pi*m/8 - 3*pi*m**2/128 - 5*pi*m**3/512 + O(m**4)
    >>> elliptic_e(1 + I, 2 - I/2).n()
    1.55203744279187 + 0.290764986058437*I
    >>> elliptic_e(0)
    pi/2
    >>> elliptic_e(2.0 - I)
    0.991052601328069 + 0.81879421395609*I

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Elliptic_integrals
    .. [2] https://functions.wolfram.com/EllipticIntegrals/EllipticE2
    .. [3] https://functions.wolfram.com/EllipticIntegrals/EllipticE

    """

    @classmethod
    def eval(cls, m, z=None):
        if z is not None:
            z, m = (m, z)
            k = 2 * z / pi
            if m.is_zero:
                return z
            if z.is_zero:
                return S.Zero
            elif k.is_integer:
                return k * elliptic_e(m)
            elif m in (S.Infinity, S.NegativeInfinity):
                return S.ComplexInfinity
            elif z.could_extract_minus_sign():
                return -elliptic_e(-z, m)
        elif m.is_zero:
            return pi / 2
        elif m is S.One:
            return S.One
        elif m is S.Infinity:
            return I * S.Infinity
        elif m is S.NegativeInfinity:
            return S.Infinity
        elif m is S.ComplexInfinity:
            return S.ComplexInfinity

    def fdiff(self, argindex=1):
        if len(self.args) == 2:
            z, m = self.args
            if argindex == 1:
                return sqrt(1 - m * sin(z) ** 2)
            elif argindex == 2:
                return (elliptic_e(z, m) - elliptic_f(z, m)) / (2 * m)
        else:
            m = self.args[0]
            if argindex == 1:
                return (elliptic_e(m) - elliptic_k(m)) / (2 * m)
        raise ArgumentIndexError(self, argindex)

    def _eval_conjugate(self):
        if len(self.args) == 2:
            z, m = self.args
            if (m.is_real and (m - 1).is_positive) is False:
                return self.func(z.conjugate(), m.conjugate())
        else:
            m = self.args[0]
            if (m.is_real and (m - 1).is_positive) is False:
                return self.func(m.conjugate())

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.simplify import hyperexpand
        if len(self.args) == 1:
            return hyperexpand(self.rewrite(hyper)._eval_nseries(x, n=n, logx=logx))
        return super()._eval_nseries(x, n=n, logx=logx)

    def _eval_rewrite_as_hyper(self, *args, **kwargs):
        if len(args) == 1:
            m = args[0]
            return pi / 2 * hyper((Rational(-1, 2), S.Half), (S.One,), m)

    def _eval_rewrite_as_meijerg(self, *args, **kwargs):
        if len(args) == 1:
            m = args[0]
            return -meijerg(((S.Half, Rational(3, 2)), []), ((S.Zero,), (S.Zero,)), -m) / 4

    def _eval_rewrite_as_Integral(self, *args):
        from sympy.integrals.integrals import Integral
        z, m = (pi / 2, self.args[0]) if len(self.args) == 1 else self.args
        t = Dummy('t')
        return Integral(sqrt(1 - m * sin(t) ** 2), (t, 0, z))