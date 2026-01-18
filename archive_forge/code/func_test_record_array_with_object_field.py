import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
def test_record_array_with_object_field():
    y = ma.masked_array([(1, '2'), (3, '4')], mask=[(0, 0), (0, 1)], dtype=[('a', int), ('b', object)])
    y[1]