import pytest
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
def test_scalar_convert_boolean():
    assert 'logical' == rinterface.baseenv['typeof'](True)[0]