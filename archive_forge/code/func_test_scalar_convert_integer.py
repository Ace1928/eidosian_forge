import pytest
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
def test_scalar_convert_integer():
    assert 'integer' == rinterface.baseenv['typeof'](int(1))[0]