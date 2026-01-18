import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_summarize_structure(self):
    A = np.arange(2002, dtype='<i8').reshape(2, 1001).view([('i', '<i8', (1001,))])
    strA = '[[([   0,    1,    2, ...,  998,  999, 1000],)]\n [([1001, 1002, 1003, ..., 1999, 2000, 2001],)]]'
    assert_equal(str(A), strA)
    reprA = "array([[([   0,    1,    2, ...,  998,  999, 1000],)],\n       [([1001, 1002, 1003, ..., 1999, 2000, 2001],)]],\n      dtype=[('i', '<i8', (1001,))])"
    assert_equal(repr(A), reprA)
    B = np.ones(2002, dtype='>i8').view([('i', '>i8', (2, 1001))])
    strB = '[([[1, 1, 1, ..., 1, 1, 1], [1, 1, 1, ..., 1, 1, 1]],)]'
    assert_equal(str(B), strB)
    reprB = "array([([[1, 1, 1, ..., 1, 1, 1], [1, 1, 1, ..., 1, 1, 1]],)],\n      dtype=[('i', '>i8', (2, 1001))])"
    assert_equal(repr(B), reprB)
    C = np.arange(22, dtype='<i8').reshape(2, 11).view([('i1', '<i8'), ('i10', '<i8', (10,))])
    strC = '[[( 0, [ 1, ..., 10])]\n [(11, [12, ..., 21])]]'
    assert_equal(np.array2string(C, threshold=1, edgeitems=1), strC)