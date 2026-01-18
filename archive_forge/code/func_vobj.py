import sys
import warnings
from string import ascii_lowercase, ascii_uppercase
import unicodedata
from sympy.printing.conventions import split_super_sub
from sympy.core.alphabets import greeks
from sympy.utilities.exceptions import sympy_deprecation_warning
def vobj(symb, height):
    """Construct vertical object of a given height

       see: xobj
    """
    return '\n'.join(xobj(symb, height))