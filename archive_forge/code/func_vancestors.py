import re
import sys
import inspect
import operator
import itertools
import collections
from inspect import getfullargspec
def vancestors(*types):
    """
            Get a list of sets of virtual ancestors for the given types
            """
    check(types)
    ras = [[] for _ in range(len(dispatch_args))]
    for types_ in typemap:
        for t, type_, ra in zip(types, types_, ras):
            if issubclass(t, type_) and type_ not in t.__mro__:
                append(type_, ra)
    return [set(ra) for ra in ras]