from __future__ import annotations
import re
from fractions import Fraction
def unicodeify(formula: str) -> str:
    """Generates a formula with unicode subscripts, e.g. Fe2O3 is transformed
    to Fe₂O₃. Does not support formulae with decimal points.

    Note that Composition now has a to_unicode_string() method that may
    be used instead.

    Args:
        formula: The string to format.
    """
    if '.' in formula:
        raise ValueError('No unicode character exists for subscript period.')
    for original_subscript, subscript_unicode in SUBSCRIPT_UNICODE.items():
        formula = formula.replace(str(original_subscript), subscript_unicode)
    return formula