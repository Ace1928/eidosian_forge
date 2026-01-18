import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
def test_vector_complex(self):
    l = [1j, 2j, 3j]
    c = numpy.array(l, dtype=numpy.complex_)
    c_r = self.check_homogeneous(c, 'complex', 'complex')
    for orig, conv in zip(l, c_r):
        assert abs(orig.real - conv.real) < 1e-06
        assert abs(orig.imag - conv.imag) < 1e-06