import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
def test_withnames(self):
    x = mrecarray(1, formats=float, names='base')
    x[0]['base'] = 10
    assert_equal(x['base'][0], 10)