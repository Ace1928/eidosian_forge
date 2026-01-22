from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import operator
import warnings
from functools import reduce
import numpy as np
from numba.np.ufunc.ufuncbuilder import _BaseUFuncBuilder, parse_identity
from numba.core import types, sigutils
from numba.core.typing import signature
from numba.np.ufunc.sigparse import parse_signature
class GUFuncSchedule(object):

    def __init__(self, parent, ishapes, oshapes, loopdims, pinned):
        self.parent = parent
        self.ishapes = ishapes
        self.oshapes = oshapes
        self.loopdims = loopdims
        self.loopn = reduce(operator.mul, loopdims, 1)
        self.pinned = pinned
        self.output_shapes = [loopdims + s for s in oshapes]

    def __str__(self):
        import pprint
        attrs = ('ishapes', 'oshapes', 'loopdims', 'loopn', 'pinned')
        values = [(k, getattr(self, k)) for k in attrs]
        return pprint.pformat(dict(values))