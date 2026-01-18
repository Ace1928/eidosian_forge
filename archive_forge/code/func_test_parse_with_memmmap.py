import shutil
import sys
import warnings
from os.path import basename, dirname
from os.path import join as pjoin
from unittest import mock
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from ...loadsave import load, save
from ...nifti1 import xform_codes
from ...testing import clear_and_catch_warnings, suppress_warnings
from ...tmpdirs import InTemporaryDirectory
from .. import gifti as gi
from ..parse_gifti_fast import GiftiImageParser, GiftiParseError
from ..util import gifti_endian_codes
def test_parse_with_memmmap():
    img1 = load(DATA_FILE7)
    img2 = load(DATA_FILE7, mmap=True)
    img3 = load(DATA_FILE7, mmap=False)
    assert len(img1.darrays) == len(img2.darrays) == 2
    assert isinstance(img1.darrays[0].data, np.memmap)
    assert isinstance(img1.darrays[1].data, np.memmap)
    assert isinstance(img2.darrays[0].data, np.memmap)
    assert isinstance(img2.darrays[1].data, np.memmap)
    assert not isinstance(img3.darrays[0].data, np.memmap)
    assert not isinstance(img3.darrays[1].data, np.memmap)
    assert_array_almost_equal(img1.darrays[0].data, DATA_FILE7_darr1)
    assert_array_almost_equal(img1.darrays[1].data, DATA_FILE7_darr2)
    assert_array_almost_equal(img2.darrays[0].data, DATA_FILE7_darr1)
    assert_array_almost_equal(img2.darrays[1].data, DATA_FILE7_darr2)
    assert_array_almost_equal(img3.darrays[0].data, DATA_FILE7_darr1)
    assert_array_almost_equal(img3.darrays[1].data, DATA_FILE7_darr2)