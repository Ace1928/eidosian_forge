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
class ImageScalingMixin:

    def assert_scaling_equal(self, hdr, slope, inter):
        h_slope, h_inter = self._get_raw_scaling(hdr)
        assert_array_equal(h_slope, slope)
        assert_array_equal(h_inter, inter)

    def assert_scale_me_scaling(self, hdr):
        slope, inter = self._get_raw_scaling(hdr)
        if not slope is None:
            assert np.isnan(slope)
        if not inter is None:
            assert np.isnan(inter)

    def _get_raw_scaling(self, hdr):
        return (hdr['scl_slope'], None)

    def _set_raw_scaling(self, hdr, slope, inter):
        hdr['scl_slope'] = slope
        if not inter is None:
            raise ValueError('inter should be None')

    def assert_null_scaling(self, arr, slope, inter):
        img_class = self.image_class
        input_hdr = img_class.header_class()
        self._set_raw_scaling(input_hdr, slope, inter)
        img = img_class(arr, np.eye(4), input_hdr)
        img_hdr = img.header
        self._set_raw_scaling(input_hdr, slope, inter)
        assert_array_equal(img.get_fdata(), arr)
        fm = bytesio_filemap(img)
        img_fobj = fm['image'].fileobj
        hdr_fobj = img_fobj if not 'header' in fm else fm['header'].fileobj
        img_hdr.write_to(hdr_fobj)
        img_hdr.data_to_fileobj(arr, img_fobj, rescale=False)
        raw_rt_img = img_class.from_file_map(fm)
        assert_array_equal(raw_rt_img.get_fdata(), arr)
        fm = bytesio_filemap(img)
        img.to_file_map(fm)
        rt_img = img_class.from_file_map(fm)
        assert_array_equal(rt_img.get_fdata(), arr)

    def test_header_scaling(self):
        img_class = self.image_class
        hdr_class = img_class.header_class
        if not hdr_class.has_data_slope:
            return
        arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        invalid_slopes = (0, np.nan, np.inf, -np.inf)
        for slope in (1,) + invalid_slopes:
            self.assert_null_scaling(arr, slope, None)
        if not hdr_class.has_data_intercept:
            return
        invalid_inters = (np.nan, np.inf, -np.inf)
        invalid_pairs = tuple(itertools.product(invalid_slopes, invalid_inters))
        bad_slopes_good_inter = tuple(itertools.product(invalid_slopes, (0, 1)))
        good_slope_bad_inters = tuple(itertools.product((1, 2), invalid_inters))
        for slope, inter in invalid_pairs + bad_slopes_good_inter + good_slope_bad_inters:
            self.assert_null_scaling(arr, slope, inter)

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

    def test_int_int_scaling(self):
        img_class = self.image_class
        arr = np.array([-1, 0, 256], dtype=np.int16)[:, None, None]
        img = img_class(arr, np.eye(4))
        hdr = img.header
        img.set_data_dtype(np.uint8)
        self._set_raw_scaling(hdr, 1, 0 if hdr.has_data_intercept else None)
        img_rt = bytesio_round_trip(img)
        assert_array_equal(img_rt.get_fdata(), np.clip(arr, 0, 255))

    @pytest.mark.parametrize('in_dtype', FLOAT_TYPES + IUINT_TYPES)
    def test_no_scaling(self, in_dtype, supported_dtype):
        img_class = self.image_class
        hdr_class = img_class.header_class
        hdr = hdr_class()
        slope = 2
        inter = 10 if hdr.has_data_intercept else 0
        mn_in, mx_in = _dt_min_max(in_dtype)
        mn = -1 if np.dtype(in_dtype).kind != 'u' else 0
        arr = np.array([mn_in, mn, 0, 1, 10, mx_in], dtype=in_dtype)
        img = img_class(arr, np.eye(4), hdr)
        img.set_data_dtype(supported_dtype)
        img.header.set_slope_inter(slope, inter)
        with np.errstate(invalid='ignore'):
            rt_img = bytesio_round_trip(img)
        with suppress_warnings():
            back_arr = np.asanyarray(rt_img.dataobj)
        exp_back = arr.copy()
        if supported_dtype in IUINT_TYPES:
            if in_dtype in FLOAT_TYPES:
                exp_back = exp_back.astype(float)
                with np.errstate(invalid='ignore'):
                    exp_back = np.round(exp_back)
                if in_dtype in FLOAT_TYPES:
                    exp_back = np.clip(exp_back, *shared_range(float, supported_dtype))
            else:
                mn_out, mx_out = _dt_min_max(supported_dtype)
                if (mn_in, mx_in) != (mn_out, mx_out):
                    exp_back = np.clip(exp_back, max(mn_in, mn_out), min(mx_in, mx_out))
        if supported_dtype in COMPLEX_TYPES:
            exp_back = exp_back.astype(supported_dtype)
        else:
            exp_back = exp_back.astype(float)
        with suppress_warnings():
            assert_allclose_safely(back_arr, exp_back * slope + inter)

    def test_write_scaling(self):
        for slope, inter, e_slope, e_inter in ((1, None, 1, None), (0, None, 1, None), (np.inf, None, 1, None), (2, None, 2, None)):
            self._check_write_scaling(slope, inter, e_slope, e_inter)

    def test_nan2zero_range_ok(self):
        img_class = self.image_class
        arr = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
        arr[0, 0, 0] = np.nan
        arr[1, 0, 0] = 256
        img = img_class(arr, np.eye(4))
        rt_img = bytesio_round_trip(img)
        assert_array_equal(rt_img.get_fdata(), arr)
        img.set_data_dtype(np.uint8)
        with np.errstate(invalid='ignore'):
            rt_img = bytesio_round_trip(img)
        assert rt_img.get_fdata()[0, 0, 0] == 0