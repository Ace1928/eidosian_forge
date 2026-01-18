import copy
import operator
import sys
import unittest
import warnings
from collections import defaultdict
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ...testing import assert_arrays_equal, clear_and_catch_warnings
from .. import tractogram as module_tractogram
from ..tractogram import (
def test_tractogram_extend(self):
    t = DATA['tractogram'].copy()
    for op, in_place in ((operator.add, False), (operator.iadd, True), (extender, True)):
        first_arg = t.copy()
        new_t = op(first_arg, t)
        assert (new_t is first_arg) == in_place
        assert_tractogram_equal(new_t[:len(t)], DATA['tractogram'])
        assert_tractogram_equal(new_t[len(t):], DATA['tractogram'])
    t = Tractogram()
    t += DATA['tractogram']
    assert_tractogram_equal(t, DATA['tractogram'])
    t = DATA['tractogram'].copy()
    t += Tractogram()
    assert_tractogram_equal(t, DATA['tractogram'])