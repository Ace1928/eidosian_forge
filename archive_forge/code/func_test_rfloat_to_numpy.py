import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
def test_rfloat_to_numpy(self):
    with (robjects.default_converter + rpyn.converter).context() as cv:
        a = robjects.r('c(1.0, 2.0, 3.0)')
    assert isinstance(a, numpy.ndarray)