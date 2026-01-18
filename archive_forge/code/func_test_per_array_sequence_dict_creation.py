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
def test_per_array_sequence_dict_creation(self):
    total_nb_rows = DATA['tractogram'].streamlines.total_nb_rows
    data_per_point = DATA['tractogram'].data_per_point
    data_dict = PerArraySequenceDict(total_nb_rows, data_per_point)
    assert data_dict.keys() == data_per_point.keys()
    for k in data_dict.keys():
        assert_arrays_equal(data_dict[k], data_per_point[k])
    del data_dict['fa']
    assert len(data_dict) == len(data_per_point) - 1
    data_per_point = DATA['data_per_point']
    data_dict = PerArraySequenceDict(total_nb_rows, data_per_point)
    assert data_dict.keys() == data_per_point.keys()
    for k in data_dict.keys():
        assert_arrays_equal(data_dict[k], data_per_point[k])
    del data_dict['fa']
    assert len(data_dict) == len(data_per_point) - 1
    data_per_point = DATA['data_per_point']
    data_dict = PerArraySequenceDict(total_nb_rows, **data_per_point)
    assert data_dict.keys() == data_per_point.keys()
    for k in data_dict.keys():
        assert_arrays_equal(data_dict[k], data_per_point[k])
    del data_dict['fa']
    assert len(data_dict) == len(data_per_point) - 1