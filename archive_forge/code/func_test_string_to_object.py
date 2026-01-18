import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def test_string_to_object(self):
    """Make sure that string-to-object functions are properly recognized"""
    old_mapper = StringConverter._mapper[:]
    conv = StringConverter(_bytes_to_date)
    assert_equal(conv._mapper, old_mapper)
    assert_(hasattr(conv, 'default'))