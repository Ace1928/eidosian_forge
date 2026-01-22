import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
class MyThing:
    __array_priority__ = 1000
    rmul_count = 0
    getitem_count = 0

    def __init__(self, shape):
        self.shape = shape

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, i):
        MyThing.getitem_count += 1
        if not isinstance(i, tuple):
            i = (i,)
        if len(i) > self.ndim:
            raise IndexError('boo')
        return MyThing(self.shape[len(i):])

    def __rmul__(self, other):
        MyThing.rmul_count += 1
        return self