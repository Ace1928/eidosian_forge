from glob import glob
from os.path import basename, dirname
from os.path import join as pjoin
from warnings import simplefilter
import numpy as np
import pytest
from numpy import array as npa
from numpy.testing import assert_almost_equal, assert_array_equal
from .. import load as top_load
from .. import parrec
from ..fileholders import FileHolder
from ..nifti1 import Nifti1Extension, Nifti1Header, Nifti1Image
from ..openers import ImageOpener
from ..parrec import (
from ..testing import assert_arr_dict_equal, clear_and_catch_warnings, suppress_warnings
from ..volumeutils import array_from_file
from . import test_spatialimages as tsi
from .test_arrayproxy import check_mmap
def test_vol_is_full():
    assert_array_equal(vol_is_full([3, 2, 1], 3), True)
    assert_array_equal(vol_is_full([3, 2, 1], 4), False)
    assert_array_equal(vol_is_full([4, 2, 1], 4), False)
    assert_array_equal(vol_is_full([3, 2, 4, 1], 4), True)
    assert_array_equal(vol_is_full([3, 2, 1], 3, 0), False)
    assert_array_equal(vol_is_full([3, 2, 0, 1], 3, 0), True)
    with pytest.raises(ValueError):
        vol_is_full([2, 1, 0], 2)
    with pytest.raises(ValueError):
        vol_is_full([3, 2, 1], 3, 2)
    assert_array_equal(vol_is_full([3, 2, 1, 2, 3, 1], 3), [True] * 6)
    assert_array_equal(vol_is_full([3, 2, 1, 2, 3], 3), [True, True, True, False, False])