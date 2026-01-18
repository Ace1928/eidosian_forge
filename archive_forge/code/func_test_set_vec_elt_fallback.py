import pytest
from rpy2.rinterface_lib import openrlib
import rpy2.rinterface
@pytest.mark.parametrize('rcls,value,func,getter', ((rpy2.rinterface.IntSexpVector, [1, 2, 3], openrlib._set_integer_elt_fallback, openrlib._get_integer_elt_fallback), (rpy2.rinterface.BoolSexpVector, [True, True, False], openrlib._set_logical_elt_fallback, openrlib._get_logical_elt_fallback), (rpy2.rinterface.FloatSexpVector, [1.1, 2.2, 3.3], openrlib._set_real_elt_fallback, openrlib._get_real_elt_fallback)))
def test_set_vec_elt_fallback(rcls, value, func, getter):
    rpy2.rinterface.initr()
    v = rcls(value)
    func(v.__sexp__._cdata, 1, value[2])
    assert getter(v.__sexp__._cdata, 1) == value[2]