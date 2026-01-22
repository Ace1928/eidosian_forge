import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def _defaultExceptionDebugAction(instring, loc, expr, exc):
    print('Exception raised:' + _ustr(exc))