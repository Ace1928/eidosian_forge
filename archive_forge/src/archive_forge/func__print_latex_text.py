from sympy.external.importtools import version_tuple
from io import BytesIO
from sympy.printing.latex import latex as default_latex
from sympy.printing.preview import preview
from sympy.utilities.misc import debug
from sympy.printing.defaults import Printable
def _print_latex_text(o):
    """
        A function to generate the latex representation of SymPy expressions.
        """
    if _can_print(o):
        s = latex(o, mode=latex_mode, **settings)
        if latex_mode == 'plain':
            return '$\\displaystyle %s$' % s
        return s