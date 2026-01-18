from sympy.core.function import Derivative
from sympy.core.function import UndefinedFunction, AppliedUndef
from sympy.core.symbol import Symbol
from sympy.interactive.printing import init_printing
from sympy.printing.latex import LatexPrinter
from sympy.printing.pretty.pretty import PrettyPrinter
from sympy.printing.pretty.pretty_symbology import center_accent
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import PRECEDENCE
def vprint(expr, **settings):
    """Function for printing of expressions generated in the
    sympy.physics vector package.

    Extends SymPy's StrPrinter, takes the same setting accepted by SymPy's
    :func:`~.sstr`, and is equivalent to ``print(sstr(foo))``.

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to print.
    settings : args
        Same as the settings accepted by SymPy's sstr().

    Examples
    ========

    >>> from sympy.physics.vector import vprint, dynamicsymbols
    >>> u1 = dynamicsymbols('u1')
    >>> print(u1)
    u1(t)
    >>> vprint(u1)
    u1

    """
    outstr = vsprint(expr, **settings)
    import builtins
    if outstr != 'None':
        builtins._ = outstr
        print(outstr)