from __future__ import annotations
import re
from fractions import Fraction
def unicodeify_spacegroup(spacegroup_symbol):
    """Generates a unicode formatted spacegroup. E.g., P2$_{1}$/c is converted to
    P2₁/c and P$\\\\overline{1}$ is converted to P̅1.

    Note that SymmetryGroup now has a to_unicode_string() method that
    may be called instead.

    Args:
        spacegroup_symbol (str): A spacegroup symbol as LaTeX

    Returns:
        A unicode spacegroup with proper subscripts and overlines.
    """
    if not spacegroup_symbol:
        return ''
    symbol = latexify_spacegroup(spacegroup_symbol)
    for num, unicode_number in SUBSCRIPT_UNICODE.items():
        symbol = symbol.replace(f'$_{{{num}}}$', unicode_number)
        symbol = symbol.replace(f'_{num}', unicode_number)
    overline = '̅'
    symbol = symbol.replace('$\\overline{', '')
    symbol = symbol.replace('$', '')
    symbol = symbol.replace('{', '')
    return symbol.replace('}', overline)