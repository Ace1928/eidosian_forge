import array
import cmath
from functools import reduce
import itertools
from operator import mul
import math
import symengine as se
from symengine.test_utilities import raises
from symengine import have_numpy
import unittest
from unittest.case import SkipTest
@unittest.skipUnless(have_numpy, 'Numpy not installed')
def test_numpy_array_out_exceptions():
    args, exprs, inp, check = _get_array()
    assert len(args) == 3 and len(exprs) == 2
    lmb = se.Lambdify(args, exprs)
    all_right = np.empty(len(exprs))
    lmb(inp, out=all_right)
    too_short = np.empty(len(exprs) - 1)
    raises(ValueError, lambda: lmb(inp, out=too_short))
    wrong_dtype = np.empty(len(exprs), dtype=int)
    raises(ValueError, lambda: lmb(inp, out=wrong_dtype))
    read_only = np.empty(len(exprs))
    read_only.flags['WRITEABLE'] = False
    raises(ValueError, lambda: lmb(inp, out=read_only))
    all_right_broadcast_C = np.empty((4, len(exprs)), order='C')
    inp_bcast = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    lmb(np.array(inp_bcast), out=all_right_broadcast_C)
    noncontig_broadcast = np.empty((4, len(exprs), 3)).transpose((1, 2, 0))
    raises(ValueError, lambda: lmb(inp_bcast, out=noncontig_broadcast))
    all_right_broadcast_F = np.empty((len(exprs), 4), order='F')
    lmb.order = 'F'
    lmb(np.array(np.array(inp_bcast).T), out=all_right_broadcast_F)