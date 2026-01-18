import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.io._harwell_boeing import (
def test_wrong_formats(self):

    def _test_invalid(bad_format):
        assert_raises(BadFortranFormat, lambda: self.parser.parse(bad_format))
    _test_invalid('I4')
    _test_invalid('(E4)')
    _test_invalid('(E4.)')
    _test_invalid('(E4.E3)')