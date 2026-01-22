import re
import sys
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_warns, IS_PYPY
class TestByteorderConverter(StringConverterTestCase):
    """ Tests of PyArray_ByteorderConverter """
    conv = mt.run_byteorder_converter
    warn = False

    def test_valid(self):
        for s in ['big', '>']:
            self._check(s, 'NPY_BIG')
        for s in ['little', '<']:
            self._check(s, 'NPY_LITTLE')
        for s in ['native', '=']:
            self._check(s, 'NPY_NATIVE')
        for s in ['ignore', '|']:
            self._check(s, 'NPY_IGNORE')
        for s in ['swap']:
            self._check(s, 'NPY_SWAP')