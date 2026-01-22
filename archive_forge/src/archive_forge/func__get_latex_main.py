import os
from os.path import join
import shutil
import tempfile
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.misc import debug
from .latex import latex
def _get_latex_main(expr, *, preamble=None, packages=(), extra_preamble=None, euler=True, fontsize=None, **latex_settings):
    """
    Generate string of a LaTeX document rendering ``expr``.
    """
    if preamble is None:
        actual_packages = packages + ('amsmath', 'amsfonts')
        if euler:
            actual_packages += ('euler',)
        package_includes = '\n' + '\n'.join(['\\usepackage{%s}' % p for p in actual_packages])
        if extra_preamble:
            package_includes += extra_preamble
        if not fontsize:
            fontsize = '12pt'
        elif isinstance(fontsize, int):
            fontsize = '{}pt'.format(fontsize)
        preamble = '\\documentclass[varwidth,%s]{standalone}\n%s\n\n\\begin{document}\n' % (fontsize, package_includes)
    elif packages or extra_preamble:
        raise ValueError('The "packages" or "extra_preamble" keywordsmust not be set if a custom LaTeX preamble was specified')
    if isinstance(expr, str):
        latex_string = expr
    else:
        latex_string = '$\\displaystyle ' + latex(expr, mode='plain', **latex_settings) + '$'
    return preamble + '\n' + latex_string + '\n\n' + '\\end{document}'