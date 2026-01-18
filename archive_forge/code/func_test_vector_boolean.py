import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
def test_vector_boolean(self):
    l = [True, False, True]
    b = numpy.array(l, dtype=numpy.bool_)
    b_r = self.check_homogeneous(b, 'logical', 'logical')
    assert tuple(l) == tuple(b_r)