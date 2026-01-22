from typing import Callable
import numpy as np
from numpy.testing import assert_array_equal, assert_, suppress_warnings
import pytest
import scipy.special as sc
def _get_ufuncs():
    ufuncs = []
    ufunc_names = []
    for name in sorted(sc.__dict__):
        obj = sc.__dict__[name]
        if not isinstance(obj, np.ufunc):
            continue
        msg = KNOWNFAILURES.get(obj)
        if msg is None:
            ufuncs.append(obj)
            ufunc_names.append(name)
        else:
            fail = pytest.mark.xfail(run=False, reason=msg)
            ufuncs.append(pytest.param(obj, marks=fail))
            ufunc_names.append(name)
    return (ufuncs, ufunc_names)