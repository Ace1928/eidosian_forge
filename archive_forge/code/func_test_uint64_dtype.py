import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def test_uint64_dtype(self):
    """Check that uint64 integer types can be specified"""
    converter = StringConverter(np.uint64, default=0)
    val = '9223372043271415339'
    assert_(converter(val) == 9223372043271415339)