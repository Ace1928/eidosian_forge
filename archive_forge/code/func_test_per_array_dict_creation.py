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
def test_per_array_dict_creation(self):
    nb_streamlines = len(DATA['tractogram'])
    data_per_streamline = DATA['tractogram'].data_per_streamline
    data_dict = PerArrayDict(nb_streamlines, data_per_streamline)
    assert data_dict.keys() == data_per_streamline.keys()
    for k in data_dict.keys():
        assert_array_equal(data_dict[k], data_per_streamline[k])
    del data_dict['mean_curvature']
    assert len(data_dict) == len(data_per_streamline) - 1
    data_per_streamline = DATA['data_per_streamline']
    data_dict = PerArrayDict(nb_streamlines, data_per_streamline)
    assert data_dict.keys() == data_per_streamline.keys()
    for k in data_dict.keys():
        assert_array_equal(data_dict[k], data_per_streamline[k])
    del data_dict['mean_curvature']
    assert len(data_dict) == len(data_per_streamline) - 1
    data_per_streamline = DATA['data_per_streamline']
    data_dict = PerArrayDict(nb_streamlines, **data_per_streamline)
    assert data_dict.keys() == data_per_streamline.keys()
    for k in data_dict.keys():
        assert_array_equal(data_dict[k], data_per_streamline[k])
    del data_dict['mean_curvature']
    assert len(data_dict) == len(data_per_streamline) - 1