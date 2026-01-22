import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class Upcase(TokenConverter):
    """Converter to upper case all matching tokens."""

    def __init__(self, *args):
        super(Upcase, self).__init__(*args)
        warnings.warn('Upcase class is deprecated, use upcaseTokens parse action instead', DeprecationWarning, stacklevel=2)

    def postParse(self, instring, loc, tokenlist):
        return list(map(str.upper, tokenlist))