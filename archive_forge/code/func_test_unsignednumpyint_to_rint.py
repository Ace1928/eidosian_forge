import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
@pytest.mark.skipif(not has_numpy, reason='package numpy cannot be imported')
@pytest.mark.parametrize('dtype', ('uint8', 'uint16'))
def test_unsignednumpyint_to_rint(dtype):
    values = (1, 2, 3)
    a = numpy.array(values, dtype=dtype)
    v = rpyn.unsignednumpyint_to_rint(a)
    assert values == tuple(v)