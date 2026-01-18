import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_unstructured_to_structured(self):
    a = np.zeros((20, 2))
    test_dtype_args = [('x', float), ('y', float)]
    test_dtype = np.dtype(test_dtype_args)
    field1 = unstructured_to_structured(a, dtype=test_dtype_args)
    field2 = unstructured_to_structured(a, dtype=test_dtype)
    assert_equal(field1, field2)