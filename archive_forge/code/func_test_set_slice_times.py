import os
import struct
import unittest
import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from nibabel import nifti1 as nifti1
from nibabel.affines import from_matvec
from nibabel.casting import have_binary128, type_info
from nibabel.eulerangles import euler2mat
from nibabel.nifti1 import (
from nibabel.optpkg import optional_package
from nibabel.pkg_info import cmp_pkg_version
from nibabel.spatialimages import HeaderDataError
from nibabel.tmpdirs import InTemporaryDirectory
from ..freesurfer import load as mghload
from ..orientations import aff2axcodes
from ..testing import (
from . import test_analyze as tana
from . import test_spm99analyze as tspm
from .nibabel_data import get_nibabel_data, needs_nibabel_data
from .test_arraywriters import IUINT_TYPES, rt_err_estimate
from .test_orientations import ALL_ORNTS
def test_set_slice_times(self):
    hdr = self.header_class()
    hdr.set_dim_info(slice=2)
    hdr.set_data_shape([1, 1, 7])
    hdr.set_slice_duration(0.1)
    times = [0] * 6
    pytest.raises(HeaderDataError, hdr.set_slice_times, times)
    times = [None] * 7
    pytest.raises(HeaderDataError, hdr.set_slice_times, times)
    times = [None, 0, 1, None, 3, 4, None]
    pytest.raises(HeaderDataError, hdr.set_slice_times, times)
    times = [None, 0, 1, 2.1, 3, 4, None]
    pytest.raises(HeaderDataError, hdr.set_slice_times, times)
    times = [None, 0, 4, 3, 2, 1, None]
    pytest.raises(HeaderDataError, hdr.set_slice_times, times)
    times = [0, 1, 2, 3, 4, 5, 6]
    hdr.set_slice_times(times)
    assert hdr['slice_code'] == 1
    assert hdr['slice_start'] == 0
    assert hdr['slice_end'] == 6
    assert hdr['slice_duration'] == 1.0
    times = [None, 0, 1, 2, 3, 4, None]
    hdr.set_slice_times(times)
    assert hdr['slice_code'] == 1
    assert hdr['slice_start'] == 1
    assert hdr['slice_end'] == 5
    assert hdr['slice_duration'] == 1.0
    times = [None, 0.4, 0.3, 0.2, 0.1, 0, None]
    hdr.set_slice_times(times)
    assert np.allclose(hdr['slice_duration'], 0.1)
    times = [None, 4, 3, 2, 1, 0, None]
    hdr.set_slice_times(times)
    assert hdr['slice_code'] == 2
    times = [None, 0, 3, 1, 4, 2, None]
    hdr.set_slice_times(times)
    assert hdr['slice_code'] == 3
    times = [None, 2, 4, 1, 3, 0, None]
    hdr.set_slice_times(times)
    assert hdr['slice_code'] == 4
    times = [None, 2, 0, 3, 1, 4, None]
    hdr.set_slice_times(times)
    assert hdr['slice_code'] == 5
    times = [None, 4, 1, 3, 0, 2, None]
    hdr.set_slice_times(times)
    assert hdr['slice_code'] == 6