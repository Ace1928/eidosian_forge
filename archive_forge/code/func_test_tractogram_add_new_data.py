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
def test_tractogram_add_new_data(self):
    t = DATA['simple_tractogram'].copy()
    t.data_per_point['fa'] = DATA['fa']
    t.data_per_point['colors'] = DATA['colors']
    t.data_per_streamline['mean_curvature'] = DATA['mean_curvature']
    t.data_per_streamline['mean_torsion'] = DATA['mean_torsion']
    t.data_per_streamline['mean_colors'] = DATA['mean_colors']
    assert_tractogram_equal(t, DATA['tractogram'])
    for i, item in enumerate(t):
        assert_tractogram_item_equal(t[i], item)
    r_tractogram = t[::-1]
    check_tractogram(r_tractogram, t.streamlines[::-1], t.data_per_streamline[::-1], t.data_per_point[::-1])
    t = Tractogram(DATA['streamlines'] * 2, affine_to_rasmm=np.eye(4))
    t = t[:len(DATA['streamlines'])]
    t.data_per_point['fa'] = DATA['fa']
    t.data_per_point['colors'] = DATA['colors']
    t.data_per_streamline['mean_curvature'] = DATA['mean_curvature']
    t.data_per_streamline['mean_torsion'] = DATA['mean_torsion']
    t.data_per_streamline['mean_colors'] = DATA['mean_colors']
    assert_tractogram_equal(t, DATA['tractogram'])