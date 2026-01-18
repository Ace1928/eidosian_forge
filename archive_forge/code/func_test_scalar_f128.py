import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
@pytest.mark.skipif(not (has_numpy and hasattr(numpy, 'float128')), reason='numpy.float128 not available on this system')
def test_scalar_f128(self):
    f128 = numpy.float128(100.000000003)
    with (robjects.default_converter + rpyn.converter).context() as cv:
        f128_r = cv.py2rpy(f128)
    f128_test = numpy.array(f128_r)[0]
    assert f128 == f128_test