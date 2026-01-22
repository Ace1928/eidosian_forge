from sympy.core import S, diff
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.logic import fuzzy_not
from sympy.core.relational import Eq, Ne
from sympy.functions.elementary.complexes import im, sign
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyroots import roots
from sympy.utilities.misc import filldedent
class Heaviside(Function):
    """
    Heaviside step function.

    Explanation
    ===========

    The Heaviside step function has the following properties:

    1) $\\frac{d}{d x} \\theta(x) = \\delta(x)$
    2) $\\theta(x) = \\begin{cases} 0 & \\text{for}\\: x < 0 \\\\ \\frac{1}{2} &
       \\text{for}\\: x = 0 \\\\1 & \\text{for}\\: x > 0 \\end{cases}$
    3) $\\frac{d}{d x} \\max(x, 0) = \\theta(x)$

    Heaviside(x) is printed as $\\theta(x)$ with the SymPy LaTeX printer.

    The value at 0 is set differently in different fields. SymPy uses 1/2,
    which is a convention from electronics and signal processing, and is
    consistent with solving improper integrals by Fourier transform and
    convolution.

    To specify a different value of Heaviside at ``x=0``, a second argument
    can be given. Using ``Heaviside(x, nan)`` gives an expression that will
    evaluate to nan for x=0.

    .. versionchanged:: 1.9 ``Heaviside(0)`` now returns 1/2 (before: undefined)

    Examples
    ========

    >>> from sympy import Heaviside, nan
    >>> from sympy.abc import x
    >>> Heaviside(9)
    1
    >>> Heaviside(-9)
    0
    >>> Heaviside(0)
    1/2
    >>> Heaviside(0, nan)
    nan
    >>> (Heaviside(x) + 1).replace(Heaviside(x), Heaviside(x, 1))
    Heaviside(x, 1) + 1

    See Also
    ========

    DiracDelta

    References
    ==========

    .. [1] https://mathworld.wolfram.com/HeavisideStepFunction.html
    .. [2] https://dlmf.nist.gov/1.16#iv

    """
    is_real = True

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of a Heaviside Function.

        Examples
        ========

        >>> from sympy import Heaviside, diff
        >>> from sympy.abc import x

        >>> Heaviside(x).fdiff()
        DiracDelta(x)

        >>> Heaviside(x**2 - 1).fdiff()
        DiracDelta(x**2 - 1)

        >>> diff(Heaviside(x)).fdiff()
        DiracDelta(x, 1)

        Parameters
        ==========

        argindex : integer
            order of derivative

        """
        if argindex == 1:
            return DiracDelta(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def __new__(cls, arg, H0=S.Half, **options):
        if isinstance(H0, Heaviside) and len(H0.args) == 1:
            H0 = S.Half
        return super(cls, cls).__new__(cls, arg, H0, **options)

    @property
    def pargs(self):
        """Args without default S.Half"""
        args = self.args
        if args[1] is S.Half:
            args = args[:1]
        return args

    @classmethod
    def eval(cls, arg, H0=S.Half):
        """
        Returns a simplified form or a value of Heaviside depending on the
        argument passed by the Heaviside object.

        Explanation
        ===========

        The ``eval()`` method is automatically called when the ``Heaviside``
        class is about to be instantiated and it returns either some simplified
        instance or the unevaluated instance depending on the argument passed.
        In other words, ``eval()`` method is not needed to be called explicitly,
        it is being called and evaluated once the object is called.

        Examples
        ========

        >>> from sympy import Heaviside, S
        >>> from sympy.abc import x

        >>> Heaviside(x)
        Heaviside(x)

        >>> Heaviside(19)
        1

        >>> Heaviside(0)
        1/2

        >>> Heaviside(0, 1)
        1

        >>> Heaviside(-5)
        0

        >>> Heaviside(S.NaN)
        nan

        >>> Heaviside(x - 100).subs(x, 5)
        0

        >>> Heaviside(x - 100).subs(x, 105)
        1

        Parameters
        ==========

        arg : argument passed by Heaviside object

        H0 : value of Heaviside(0)

        """
        if arg.is_extended_negative:
            return S.Zero
        elif arg.is_extended_positive:
            return S.One
        elif arg.is_zero:
            return H0
        elif arg is S.NaN:
            return S.NaN
        elif fuzzy_not(im(arg).is_zero):
            raise ValueError('Function defined only for Real Values. Complex part: %s  found in %s .' % (repr(im(arg)), repr(arg)))

    def _eval_rewrite_as_Piecewise(self, arg, H0=None, **kwargs):
        """
        Represents Heaviside in a Piecewise form.

        Examples
        ========

        >>> from sympy import Heaviside, Piecewise, Symbol, nan
        >>> x = Symbol('x')

        >>> Heaviside(x).rewrite(Piecewise)
        Piecewise((0, x < 0), (1/2, Eq(x, 0)), (1, True))

        >>> Heaviside(x,nan).rewrite(Piecewise)
        Piecewise((0, x < 0), (nan, Eq(x, 0)), (1, True))

        >>> Heaviside(x - 5).rewrite(Piecewise)
        Piecewise((0, x < 5), (1/2, Eq(x, 5)), (1, True))

        >>> Heaviside(x**2 - 1).rewrite(Piecewise)
        Piecewise((0, x**2 < 1), (1/2, Eq(x**2, 1)), (1, True))

        """
        if H0 == 0:
            return Piecewise((0, arg <= 0), (1, True))
        if H0 == 1:
            return Piecewise((0, arg < 0), (1, True))
        return Piecewise((0, arg < 0), (H0, Eq(arg, 0)), (1, True))

    def _eval_rewrite_as_sign(self, arg, H0=S.Half, **kwargs):
        """
        Represents the Heaviside function in the form of sign function.

        Explanation
        ===========

        The value of Heaviside(0) must be 1/2 for rewriting as sign to be
        strictly equivalent. For easier usage, we also allow this rewriting
        when Heaviside(0) is undefined.

        Examples
        ========

        >>> from sympy import Heaviside, Symbol, sign, nan
        >>> x = Symbol('x', real=True)
        >>> y = Symbol('y')

        >>> Heaviside(x).rewrite(sign)
        sign(x)/2 + 1/2

        >>> Heaviside(x, 0).rewrite(sign)
        Piecewise((sign(x)/2 + 1/2, Ne(x, 0)), (0, True))

        >>> Heaviside(x, nan).rewrite(sign)
        Piecewise((sign(x)/2 + 1/2, Ne(x, 0)), (nan, True))

        >>> Heaviside(x - 2).rewrite(sign)
        sign(x - 2)/2 + 1/2

        >>> Heaviside(x**2 - 2*x + 1).rewrite(sign)
        sign(x**2 - 2*x + 1)/2 + 1/2

        >>> Heaviside(y).rewrite(sign)
        Heaviside(y)

        >>> Heaviside(y**2 - 2*y + 1).rewrite(sign)
        Heaviside(y**2 - 2*y + 1)

        See Also
        ========

        sign

        """
        if arg.is_extended_real:
            pw1 = Piecewise(((sign(arg) + 1) / 2, Ne(arg, 0)), (Heaviside(0, H0=H0), True))
            pw2 = Piecewise(((sign(arg) + 1) / 2, Eq(Heaviside(0, H0=H0), S.Half)), (pw1, True))
            return pw2

    def _eval_rewrite_as_SingularityFunction(self, args, H0=S.Half, **kwargs):
        """
        Returns the Heaviside expression written in the form of Singularity
        Functions.

        """
        from sympy.solvers import solve
        from sympy.functions.special.singularity_functions import SingularityFunction
        if self == Heaviside(0):
            return SingularityFunction(0, 0, 0)
        free = self.free_symbols
        if len(free) == 1:
            x = free.pop()
            return SingularityFunction(x, solve(args, x)[0], 0)
        else:
            raise TypeError(filldedent('\n                rewrite(SingularityFunction) does not\n                support arguments with more that one variable.'))