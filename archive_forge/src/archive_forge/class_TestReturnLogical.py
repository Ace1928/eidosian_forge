import pytest
from numpy import array
from . import util
class TestReturnLogical(util.F2PyTest):

    def check_function(self, t):
        assert t(True) == 1
        assert t(False) == 0
        assert t(0) == 0
        assert t(None) == 0
        assert t(0.0) == 0
        assert t(0j) == 0
        assert t(1j) == 1
        assert t(234) == 1
        assert t(234.6) == 1
        assert t(234.6 + 3j) == 1
        assert t('234') == 1
        assert t('aaa') == 1
        assert t('') == 0
        assert t([]) == 0
        assert t(()) == 0
        assert t({}) == 0
        assert t(t) == 1
        assert t(-234) == 1
        assert t(10 ** 100) == 1
        assert t([234]) == 1
        assert t((234,)) == 1
        assert t(array(234)) == 1
        assert t(array([234])) == 1
        assert t(array([[234]])) == 1
        assert t(array([127], 'b')) == 1
        assert t(array([234], 'h')) == 1
        assert t(array([234], 'i')) == 1
        assert t(array([234], 'l')) == 1
        assert t(array([234], 'f')) == 1
        assert t(array([234], 'd')) == 1
        assert t(array([234 + 3j], 'F')) == 1
        assert t(array([234], 'D')) == 1
        assert t(array(0)) == 0
        assert t(array([0])) == 0
        assert t(array([[0]])) == 0
        assert t(array([0j])) == 0
        assert t(array([1])) == 1
        pytest.raises(ValueError, t, array([0, 0]))