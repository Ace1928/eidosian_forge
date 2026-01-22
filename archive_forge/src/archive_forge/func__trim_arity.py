import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def _trim_arity(func, maxargs=2):
    if func in singleArgBuiltins:
        return lambda s, l, t: func(t)
    limit = [0]
    foundArity = [False]

    def wrapper(*args):
        while 1:
            try:
                ret = func(*args[limit[0]:])
                foundArity[0] = True
                return ret
            except TypeError:
                if limit[0] <= maxargs and (not foundArity[0]):
                    limit[0] += 1
                    continue
                raise
    return wrapper