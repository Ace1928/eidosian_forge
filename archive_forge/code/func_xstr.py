import sys
import warnings
from string import ascii_lowercase, ascii_uppercase
import unicodedata
from sympy.printing.conventions import split_super_sub
from sympy.core.alphabets import greeks
from sympy.utilities.exceptions import sympy_deprecation_warning
def xstr(*args):
    sympy_deprecation_warning('\n        The sympy.printing.pretty.pretty_symbology.xstr() function is\n        deprecated. Use str() instead.\n        ', deprecated_since_version='1.7', active_deprecations_target='deprecated-pretty-printing-functions')
    return str(*args)