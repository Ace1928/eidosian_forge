from __future__ import division
import uncertainties
import uncertainties.core as uncert_core
from uncertainties import ufloat, unumpy, test_uncertainties
from uncertainties.unumpy import core
from uncertainties.test_uncertainties import numbers_close, arrays_close
def test_pseudo_inverse():
    """Tests of the pseudo-inverse"""
    pinv_num = core.wrap_array_func(numpy.linalg.pinv)
    m = unumpy.matrix([[ufloat(10, 1), -3.1], [0, ufloat(3, 0)], [1, -3.1]])
    rcond = 1e-08
    m_pinv_num = pinv_num(m, rcond)
    m_pinv_package = core.pinv(m, rcond)
    assert arrays_close(m_pinv_num, m_pinv_package)
    vector = [ufloat(10, 1), -3.1, 11]
    m = unumpy.matrix([vector, vector])
    m_pinv_num = pinv_num(m, rcond)
    m_pinv_package = core.pinv(m, rcond)
    assert arrays_close(m_pinv_num, m_pinv_package)
    m = unumpy.matrix([[ufloat(10, 1), 0], [3, 0]])
    m_pinv_num = pinv_num(m, rcond)
    m_pinv_package = core.pinv(m, rcond)
    assert arrays_close(m_pinv_num, m_pinv_package)