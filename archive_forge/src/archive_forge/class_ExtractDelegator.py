import abc
import collections.abc
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import sexp
from . import conversion
import rpy2.rlike.container as rlc
import datetime
import copy
import itertools
import math
import os
import jinja2  # type: ignore
import time
import tzlocal
from time import struct_time, mktime
import typing
import warnings
from rpy2.rinterface import (Sexp, ListSexpVector, StrSexpVector,
class ExtractDelegator(object):
    """ Delegate the R 'extraction' ("[") and 'replacement' ("[<-")
    of items in a vector
    or vector-like object. This can help making syntactic
    niceties possible."""
    _extractfunction = rinterface.baseenv['[']
    _replacefunction = rinterface.baseenv['[<-']

    def __init__(self, parent):
        self._parent = parent

    def __call__(self, *args, **kwargs):
        """ Subset the "R-way.", using R's "[" function.
           In a nutshell, R indexing differs from Python indexing on:

           - indexing can be done with integers or strings (that are 'names')

           - an index equal to TRUE will mean everything selected
             (because of the recycling rule)

           - integer indexing starts at one

           - negative integer indexing means exclusion of the given integers

           - an index is itself a vector of elements to select
        """
        conv_args = [None] * (len(args) + 1)
        conv_args[0] = self._parent
        cv = conversion.get_conversion()
        for i, x in enumerate(args, 1):
            if x is MissingArg:
                conv_args[i] = x
            else:
                conv_args[i] = cv.py2rpy(x)
        kwargs = copy.copy(kwargs)
        for k, v in kwargs.items():
            kwargs[k] = cv.py2rpy(v)
        fun = self._extractfunction
        res = fun(*conv_args, **kwargs)
        res = cv.rpy2py(res)
        return res

    def __getitem__(self, item):
        res = self(item)
        return res

    def __setitem__(self, item, value):
        """ Assign a given value to a given index position in the vector.
        The index position can either be:
        - an int: x[1] = y
        - a tuple of ints: x[1, 2, 3] = y
        - an item-able object (such as a dict): x[{'i': 1}] = y
        """
        fun = self._replacefunction
        cv = conversion.get_conversion()
        if type(item) is tuple:
            args = list([None] * (len(item) + 2))
            for i, v in enumerate(item):
                if v is MissingArg:
                    continue
                args[i + 1] = cv.py2rpy(v)
            args[-1] = cv.py2rpy(value)
            args[0] = self._parent
            res = fun(*args)
        elif type(item) is dict or type(item) is rlc.TaggedList:
            args = rlc.TaggedList.from_items(item)
            for i, (k, v) in enumerate(args.items()):
                args[i] = cv.py2rpy(v)
            args.append(cv.py2rpy(value), tag=None)
            args.insert(0, self._parent, tag=None)
            res = fun.rcall(tuple(args.items()), globalenv_ri)
        else:
            args = [self._parent, cv.py2rpy(item), cv.py2rpy(value)]
            res = fun(*args)
        self._parent.__sexp__ = res.__sexp__