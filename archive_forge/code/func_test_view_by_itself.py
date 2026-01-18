import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
def test_view_by_itself(self):
    mrec, a, b, arr = self.data
    test = mrec.view()
    assert_(isinstance(test, MaskedRecords))
    assert_equal_records(test, mrec)
    assert_equal_records(test._mask, mrec._mask)