import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_writebacks():
    a = np.arange(6, dtype='f4')
    au = a.byteswap().newbyteorder()
    assert_(a.dtype.byteorder != au.dtype.byteorder)
    it = nditer(au, [], [['readwrite', 'updateifcopy']], casting='equiv', op_dtypes=[np.dtype('f4')])
    with it:
        it.operands[0][:] = 100
    assert_equal(au, 100)
    it = nditer(au, [], [['readwrite', 'updateifcopy']], casting='equiv', op_dtypes=[np.dtype('f4')])
    try:
        with it:
            assert_equal(au.flags.writeable, False)
            it.operands[0][:] = 0
            raise ValueError('exit context manager on exception')
    except:
        pass
    assert_equal(au, 0)
    assert_equal(au.flags.writeable, True)
    assert_raises(ValueError, getattr, it, 'operands')
    it = nditer(au, [], [['readwrite', 'updateifcopy']], casting='equiv', op_dtypes=[np.dtype('f4')])
    with it:
        x = it.operands[0]
        x[:] = 6
        assert_(x.flags.writebackifcopy)
    assert_equal(au, 6)
    assert_(not x.flags.writebackifcopy)
    x[:] = 123
    assert_equal(au, 6)
    it = nditer(au, [], [['readwrite', 'updateifcopy']], casting='equiv', op_dtypes=[np.dtype('f4')])
    with it:
        with it:
            for x in it:
                x[...] = 123
    it = nditer(au, [], [['readwrite', 'updateifcopy']], casting='equiv', op_dtypes=[np.dtype('f4')])
    with it:
        with it:
            for x in it:
                x[...] = 123
        assert_raises(ValueError, getattr, it, 'operands')
    it = nditer(au, [], [['readwrite', 'updateifcopy']], casting='equiv', op_dtypes=[np.dtype('f4')])
    del au
    with it:
        for x in it:
            x[...] = 123
    enter = it.__enter__
    assert_raises(RuntimeError, enter)