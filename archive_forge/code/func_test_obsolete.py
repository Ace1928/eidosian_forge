from __future__ import division
import uncertainties
import uncertainties.core as uncert_core
from uncertainties import ufloat, unumpy, test_uncertainties
from uncertainties.unumpy import core
from uncertainties.test_uncertainties import numbers_close, arrays_close
def test_obsolete():
    """Test of obsolete functions"""
    arr_obs = unumpy.uarray.__call__(([1, 2], [1, 4]))
    arr = unumpy.uarray([1, 2], [1, 4])
    assert arrays_close(arr_obs, arr)
    mat_obs = unumpy.umatrix.__call__(([1, 2], [1, 4]))
    mat = unumpy.umatrix([1, 2], [1, 4])
    assert arrays_close(mat_obs, mat)