import itertools
import unittest
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..optpkg import optional_package
from ..casting import sctypes_aliases, shared_range, type_info
from ..spatialimages import HeaderDataError
from ..spm99analyze import HeaderTypeError, Spm99AnalyzeHeader, Spm99AnalyzeImage
from ..testing import (
from ..volumeutils import _dt_min_max, apply_read_scaling
from . import test_analyze
def _check_write_scaling(self, slope, inter, effective_slope, effective_inter):
    img_class = self.image_class
    arr = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    arr[0, 0, 0] = 0.4
    arr[1, 0, 0] = 0.6
    aff = np.eye(4)
    img = img_class(arr, aff)
    self.assert_scale_me_scaling(img.header)
    hdr = img.header
    self._set_raw_scaling(hdr, slope, inter)
    img = img_class(arr, aff)
    self.assert_scale_me_scaling(img.header)
    assert_array_equal(img.get_fdata(), arr)
    img_rt = bytesio_round_trip(img)
    self.assert_scale_me_scaling(img_rt.header)
    assert_array_equal(img_rt.get_fdata(), arr)
    self._set_raw_scaling(img.header, slope, inter)
    self.assert_scaling_equal(img.header, slope, inter)
    assert_array_equal(img.get_fdata(), arr)
    img_rt = bytesio_round_trip(img)
    assert_array_equal(img_rt.get_fdata(), apply_read_scaling(arr, effective_slope, effective_inter))
    do_slope, do_inter = img.header.get_slope_inter()
    assert_array_equal(img_rt.dataobj.slope, 1 if do_slope is None else do_slope)
    assert_array_equal(img_rt.dataobj.inter, 0 if do_inter is None else do_inter)
    self.assert_scale_me_scaling(img_rt.header)
    self.assert_scaling_equal(img.header, slope, inter)
    img.header.set_data_dtype(np.uint8)
    with np.errstate(invalid='ignore'):
        img_rt = bytesio_round_trip(img)
    assert_array_equal(img_rt.get_fdata(), apply_read_scaling(np.round(arr), effective_slope, effective_inter))
    arr[-1, -1, -1] = 256
    arr[-2, -1, -1] = -1
    with np.errstate(invalid='ignore'):
        img_rt = bytesio_round_trip(img)
    exp_unscaled_arr = np.clip(np.round(arr), 0, 255)
    assert_array_equal(img_rt.get_fdata(), apply_read_scaling(exp_unscaled_arr, effective_slope, effective_inter))