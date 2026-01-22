import re
import sys
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_warns, IS_PYPY
class TestSortkindConverter(StringConverterTestCase):
    """ Tests of PyArray_SortkindConverter """
    conv = mt.run_sortkind_converter
    warn = False

    def test_valid(self):
        self._check('quicksort', 'NPY_QUICKSORT')
        self._check('heapsort', 'NPY_HEAPSORT')
        self._check('mergesort', 'NPY_STABLESORT')
        self._check('stable', 'NPY_STABLESORT')