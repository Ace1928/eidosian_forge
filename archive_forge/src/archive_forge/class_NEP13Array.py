import numpy as np
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba import vectorize, guvectorize
from numba.np.ufunc import PyUFunc_One
from numba.np.ufunc.dufunc import DUFunc as UFuncBuilder
from numba.tests.support import tag, TestCase
from numba.core import config
import unittest
class NEP13Array:

    def __init__(self, array):
        self.array = array

    def __array__(self):
        return self.array

    def tolist(self):
        return self.array.tolist()

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method != '__call__':
            return NotImplemented
        return NEP13Array(ufunc(*[np.asarray(x) for x in args], **kwargs))