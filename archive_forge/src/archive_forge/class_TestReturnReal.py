import platform
import pytest
import numpy as np
from numpy import array
from . import util
class TestReturnReal(util.F2PyTest):

    def check_function(self, t, tname):
        if tname in ['t0', 't4', 's0', 's4']:
            err = 1e-05
        else:
            err = 0.0
        assert abs(t(234) - 234.0) <= err
        assert abs(t(234.6) - 234.6) <= err
        assert abs(t('234') - 234) <= err
        assert abs(t('234.6') - 234.6) <= err
        assert abs(t(-234) + 234) <= err
        assert abs(t([234]) - 234) <= err
        assert abs(t((234,)) - 234.0) <= err
        assert abs(t(array(234)) - 234.0) <= err
        assert abs(t(array(234).astype('b')) + 22) <= err
        assert abs(t(array(234, 'h')) - 234.0) <= err
        assert abs(t(array(234, 'i')) - 234.0) <= err
        assert abs(t(array(234, 'l')) - 234.0) <= err
        assert abs(t(array(234, 'B')) - 234.0) <= err
        assert abs(t(array(234, 'f')) - 234.0) <= err
        assert abs(t(array(234, 'd')) - 234.0) <= err
        if tname in ['t0', 't4', 's0', 's4']:
            assert t(1e+200) == t(1e+300)
        pytest.raises(ValueError, t, 'abc')
        pytest.raises(IndexError, t, [])
        pytest.raises(IndexError, t, ())
        pytest.raises(Exception, t, t)
        pytest.raises(Exception, t, {})
        try:
            r = t(10 ** 400)
            assert repr(r) in ['inf', 'Infinity']
        except OverflowError:
            pass