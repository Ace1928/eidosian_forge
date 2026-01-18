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
def test_tractogram_getitem(self):
    for i, t in enumerate(DATA['tractogram']):
        assert_tractogram_item_equal(DATA['tractogram'][i], t)
    tractogram_view = DATA['simple_tractogram'][::2]
    check_tractogram(tractogram_view, DATA['streamlines'][::2])
    r_tractogram = DATA['tractogram'][::-1]
    check_tractogram(r_tractogram, DATA['streamlines'][::-1], DATA['tractogram'].data_per_streamline[::-1], DATA['tractogram'].data_per_point[::-1])
    tractogram = DATA['tractogram'].copy()
    tractogram.affine_to_rasmm = DATA['rng'].rand(4, 4)
    tractogram_view = tractogram[::2]
    assert_array_equal(tractogram_view.affine_to_rasmm, tractogram.affine_to_rasmm)