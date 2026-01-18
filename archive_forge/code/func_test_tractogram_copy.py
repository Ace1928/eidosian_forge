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
def test_tractogram_copy(self):
    tractogram = DATA['tractogram'].copy()
    assert tractogram is not DATA['tractogram']
    assert tractogram.streamlines is not DATA['tractogram'].streamlines
    assert tractogram.data_per_streamline is not DATA['tractogram'].data_per_streamline
    assert tractogram.data_per_point is not DATA['tractogram'].data_per_point
    for key in tractogram.data_per_streamline:
        assert tractogram.data_per_streamline[key] is not DATA['tractogram'].data_per_streamline[key]
    for key in tractogram.data_per_point:
        assert tractogram.data_per_point[key] is not DATA['tractogram'].data_per_point[key]
    assert_tractogram_equal(tractogram, DATA['tractogram'])