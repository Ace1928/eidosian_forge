import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def _defaultStartDebugAction(instring, loc, expr):
    print('Match ' + _ustr(expr) + ' at loc ' + _ustr(loc) + '(%d,%d)' % (lineno(loc, instring), col(loc, instring)))