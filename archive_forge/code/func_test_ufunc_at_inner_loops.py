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
@pytest.mark.parametrize('typecode', np.typecodes['AllInteger'] + np.typecodes['Float'])
@pytest.mark.parametrize('ufunc', indexed_ufuncs)
def test_ufunc_at_inner_loops(self, typecode, ufunc):
    if ufunc is np.divide and typecode in np.typecodes['AllInteger']:
        a = np.ones(100, dtype=typecode)
        indx = np.random.randint(100, size=30, dtype=np.intp)
        vals = np.arange(1, 31, dtype=typecode)
    else:
        a = np.ones(1000, dtype=typecode)
        indx = np.random.randint(1000, size=3000, dtype=np.intp)
        vals = np.arange(3000, dtype=typecode)
    atag = a.copy()
    with warnings.catch_warnings(record=True) as w_at:
        warnings.simplefilter('always')
        ufunc.at(a, indx, vals)
    with warnings.catch_warnings(record=True) as w_loop:
        warnings.simplefilter('always')
        for i, v in zip(indx, vals):
            ufunc(atag[i], v, out=atag[i:i + 1], casting='unsafe')
    assert_equal(atag, a)
    if len(w_loop) > 0:
        assert len(w_at) > 0
        assert w_at[0].category == w_loop[0].category
        assert str(w_at[0].message)[:10] == str(w_loop[0].message)[:10]