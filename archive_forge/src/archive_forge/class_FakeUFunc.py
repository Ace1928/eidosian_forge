import sys
from itertools import product
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaNotImplementedError
from numba.tests.support import TestCase
from numba.tests.enum_usecases import Shake, RequestError
from numba.np import numpy_support
class FakeUFunc(object):
    __slots__ = ('nin', 'nout', 'types', 'ntypes')
    __name__ = 'fake ufunc'

    def __init__(self, types):
        self.types = types
        in_, out = self.types[0].split('->')
        self.nin = len(in_)
        self.nout = len(out)
        self.ntypes = len(types)
        for tp in types:
            in_, out = self.types[0].split('->')
            assert len(in_) == self.nin
            assert len(out) == self.nout