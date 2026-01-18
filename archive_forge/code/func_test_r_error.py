import pytest
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
def test_r_error():
    r_sum = rinterface.baseenv['sum']
    letters = rinterface.baseenv['letters']
    with pytest.raises(rinterface.embedded.RRuntimeError), pytest.warns(rinterface.RRuntimeWarning):
        r_sum(letters)