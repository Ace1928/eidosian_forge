from sympy.external.importtools import version_tuple
from io import BytesIO
from sympy.printing.latex import latex as default_latex
from sympy.printing.preview import preview
from sympy.utilities.misc import debug
from sympy.printing.defaults import Printable
def _svg_wrapper(o):
    exprbuffer = BytesIO()
    try:
        preview(o, output='svg', viewer='BytesIO', euler=euler, outputbuffer=exprbuffer, extra_preamble=extra_preamble, dvioptions=dvioptions_svg, fontsize=fontsize)
    except Exception as e:
        debug('svg printing:', '_preview_wrapper exception raised:', repr(e))
        raise
    return exprbuffer.getvalue().decode('utf-8')