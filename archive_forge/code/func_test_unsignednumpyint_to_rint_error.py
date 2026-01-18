import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
@pytest.mark.skipif(not has_numpy, reason='package numpy cannot be imported')
@pytest.mark.parametrize('dtype', ('uint32', 'uint64'))
def test_unsignednumpyint_to_rint_error(dtype):
    values = (1, 2, 3)
    a = numpy.array(values, dtype=dtype)
    with pytest.raises(ValueError):
        rpyn.unsignednumpyint_to_rint(a)