from sympy.external.importtools import version_tuple
from io import BytesIO
from sympy.printing.latex import latex as default_latex
from sympy.printing.preview import preview
from sympy.utilities.misc import debug
from sympy.printing.defaults import Printable
def _print_plain(arg, p, cycle):
    """caller for pretty, for use in IPython 0.11"""
    if _can_print(arg):
        p.text(stringify_func(arg))
    else:
        p.text(IPython.lib.pretty.pretty(arg))