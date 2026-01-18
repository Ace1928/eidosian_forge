import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def upcaseTokens(s, l, t):
    """Helper parse action to convert tokens to upper case."""
    return [tt.upper() for tt in map(_ustr, t)]