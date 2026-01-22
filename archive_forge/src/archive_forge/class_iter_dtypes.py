import operator as op
from functools import partial
import sys
import numpy as np
from .. import units as pq
from ..quantity import Quantity
from .common import TestCase
class iter_dtypes:

    def __init__(self):
        self._i = 1
        self._typeDict = np.sctypeDict.copy()
        self._typeDict[17] = int
        self._typeDict[18] = long
        self._typeDict[19] = float
        self._typeDict[20] = complex

    def __iter__(self):
        return self

    def __next__(self):
        if self._i > 20:
            raise StopIteration
        i = self._i
        self._i += 1
        return self._typeDict[i]

    def next(self):
        return self.__next__()