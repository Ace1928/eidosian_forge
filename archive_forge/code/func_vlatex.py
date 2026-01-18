from sympy.core.function import Derivative
from sympy.core.function import UndefinedFunction, AppliedUndef
from sympy.core.symbol import Symbol
from sympy.interactive.printing import init_printing
from sympy.printing.latex import LatexPrinter
from sympy.printing.pretty.pretty import PrettyPrinter
from sympy.printing.pretty.pretty_symbology import center_accent
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import PRECEDENCE
def vlatex(expr, **settings):
    """Function for printing latex representation of sympy.physics.vector
    objects.

    For latex representation of Vectors, Dyadics, and dynamicsymbols. Takes the
    same options as SymPy's :func:`~.latex`; see that function for more
    information;

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to represent in LaTeX form
    settings : args
        Same as latex()

    Examples
    ========

    >>> from sympy.physics.vector import vlatex, ReferenceFrame, dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q1, q2 = dynamicsymbols('q1 q2')
    >>> q1d, q2d = dynamicsymbols('q1 q2', 1)
    >>> q1dd, q2dd = dynamicsymbols('q1 q2', 2)
    >>> vlatex(N.x + N.y)
    '\\\\mathbf{\\\\hat{n}_x} + \\\\mathbf{\\\\hat{n}_y}'
    >>> vlatex(q1 + q2)
    'q_{1} + q_{2}'
    >>> vlatex(q1d)
    '\\\\dot{q}_{1}'
    >>> vlatex(q1 * q2d)
    'q_{1} \\\\dot{q}_{2}'
    >>> vlatex(q1dd * q1 / q1d)
    '\\\\frac{q_{1} \\\\ddot{q}_{1}}{\\\\dot{q}_{1}}'

    """
    latex_printer = VectorLatexPrinter(settings)
    return latex_printer.doprint(expr)