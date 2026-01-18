import getpass
import hashlib
import os
import struct
import time
import unittest
from os.path import isdir
from os.path import join as pjoin
from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose
from ...fileslice import strided_scalar
from ...testing import clear_and_catch_warnings
from ...tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from ...tmpdirs import InTemporaryDirectory
from .. import (
from ..io import _pack_rgb
def test_write_morph_data():
    """Test write_morph_data edge cases"""
    values = np.arange(20, dtype='>f4')
    okay_shapes = [(20,), (20, 1), (20, 1, 1), (1, 20)]
    bad_shapes = [(10, 2), (1, 1, 20, 1, 1)]
    big_num = np.iinfo('i4').max + 1
    with InTemporaryDirectory():
        for shape in okay_shapes:
            write_morph_data('test.curv', values.reshape(shape))
            assert np.array_equal(read_morph_data('test.curv'), values)
        with pytest.raises(ValueError):
            write_morph_data('test.curv', np.zeros(shape), big_num)
        if np.dtype(int) != np.dtype(np.int32):
            with pytest.raises(ValueError):
                write_morph_data('test.curv', strided_scalar((big_num,)))
        for shape in bad_shapes:
            with pytest.raises(ValueError):
                write_morph_data('test.curv', values.reshape(shape))