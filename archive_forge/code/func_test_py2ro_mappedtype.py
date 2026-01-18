import array
import pytest
import rpy2.rinterface_lib.sexp
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import conversion
@pytest.mark.parametrize('value,cls', [(1, int), (True, bool), (b'houba', bytes), (1.0, float), (1.0 + 2j, complex)])
def test_py2ro_mappedtype(value, cls):
    pyobj = value
    assert isinstance(pyobj, cls)
    rob = robjects.default_converter.py2rpy(pyobj)
    assert isinstance(rob, cls)