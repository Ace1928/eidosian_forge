import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
@pytest.mark.skipif(not has_numpy, reason='package numpy cannot be imported')
@pytest.mark.parametrize('values,expected_cls', ((['a', 1, 2], robjects.vectors.ListVector), (['a', 'b', 'c'], rinterface.StrSexpVector), ([b'a', b'b', b'c'], rinterface.ByteSexpVector)))
def test_numpy_O_py2rpy(values, expected_cls):
    a = numpy.array(values, dtype='O')
    v = rpyn.numpy_O_py2rpy(a)
    assert isinstance(v, expected_cls)