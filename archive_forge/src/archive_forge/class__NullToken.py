import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class _NullToken(object):

    def __bool__(self):
        return False
    __nonzero__ = __bool__

    def __str__(self):
        return ''