import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def test_validate_nb_names(self):
    """Test validate nb names"""
    namelist = ('a', 'b', 'c')
    validator = NameValidator()
    assert_equal(validator(namelist, nbfields=1), ('a',))
    assert_equal(validator(namelist, nbfields=5, defaultfmt='g%i'), ['a', 'b', 'c', 'g0', 'g1'])