import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
def test_vector_integer_py2rpy(self):
    l = [1, 2, 3]
    i = numpy.array(l, dtype='i')
    i_r = self.check_homogeneous(i, 'numeric', 'integer')
    assert tuple(l) == tuple(i_r)