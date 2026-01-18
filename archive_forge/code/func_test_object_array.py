import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
def test_object_array(self):
    o = numpy.array([1, 'a', 3.2], dtype=numpy.object_)
    with (robjects.default_converter + rpyn.converter).context() as cv:
        o_r = cv.py2rpy(o)
    assert r['mode'](o_r)[0] == 'list'
    assert r['[['](o_r, 1)[0] == 1
    assert r['[['](o_r, 2)[0] == 'a'
    assert r['[['](o_r, 3)[0] == 3.2