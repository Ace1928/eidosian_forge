import itertools
import contextlib
import operator
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_raises, assert_equal
def test_to_128():
    with exc_iter(INT64_VALUES) as it:
        for a, in it:
            b = mt.extint_to_128(a)
            if a != b:
                assert_equal(b, a)