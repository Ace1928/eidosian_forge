import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
def test_record_array(self):
    rec = numpy.array([(1, 2.3), (2, -0.7), (3, 12.1)], dtype=[('count', 'i'), ('value', numpy.double)])
    with (robjects.default_converter + rpyn.converter).context() as cv:
        rec_r = cv.py2rpy(rec)
    assert r['is.data.frame'](rec_r)[0] is True
    assert tuple(r['names'](rec_r)) == ('count', 'value')
    count_r = rec_r[rec_r.names.index('count')]
    value_r = rec_r[rec_r.names.index('value')]
    assert r['storage.mode'](count_r)[0] == 'integer'
    assert r['storage.mode'](value_r)[0] == 'double'
    assert count_r[1] == 2
    assert value_r[2] == 12.1