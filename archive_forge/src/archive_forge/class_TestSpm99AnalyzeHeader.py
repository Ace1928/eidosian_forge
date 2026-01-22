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
class TestSpm99AnalyzeHeader(test_analyze.TestAnalyzeHeader, HeaderScalingMixin):
    header_class = Spm99AnalyzeHeader

    def test_empty(self):
        super().test_empty()
        hdr = self.header_class()
        assert hdr['scl_slope'] == 1

    def test_big_scaling(self):
        hdr = self.header_class()
        hdr.set_data_shape((1, 1, 1))
        hdr.set_data_dtype(np.int16)
        sio = BytesIO()
        dtt = np.float32
        data = np.array([type_info(dtt)['max']], dtype=dtt)[:, None, None]
        hdr.data_to_fileobj(data, sio)
        data_back = hdr.data_from_fileobj(sio)
        assert np.allclose(data, data_back)

    def test_slope_inter(self):
        hdr = self.header_class()
        assert hdr.get_slope_inter() == (1.0, None)
        for in_tup, exp_err, out_tup, raw_slope in (((2.0,), None, (2.0, None), 2.0), ((None,), None, (None, None), np.nan), ((1.0, None), None, (1.0, None), 1.0), ((None, 1.1), HeaderTypeError, (None, None), np.nan), ((2.0, 1.1), HeaderTypeError, (None, None), 2.0), ((0.0, None), HeaderDataError, (None, None), 0.0), ((np.nan, np.nan), None, (None, None), np.nan), ((np.nan, None), None, (None, None), np.nan), ((None, np.nan), None, (None, None), np.nan), ((np.inf, None), HeaderDataError, (None, None), np.inf), ((-np.inf, None), HeaderDataError, (None, None), -np.inf), ((None, 0.0), None, (None, None), np.nan)):
            hdr = self.header_class()
            if not exp_err is None:
                with pytest.raises(exp_err):
                    hdr.set_slope_inter(*in_tup)
                if not in_tup[0] is None:
                    hdr['scl_slope'] = in_tup[0]
            else:
                hdr.set_slope_inter(*in_tup)
                assert hdr.get_slope_inter() == out_tup
                hdr = Spm99AnalyzeHeader.from_header(hdr, check=True)
                assert hdr.get_slope_inter() == out_tup
            assert_array_equal(hdr['scl_slope'], raw_slope)

    def test_origin_checks(self):
        HC = self.header_class
        hdr = HC()
        hdr.data_shape = [1, 1, 1]
        hdr['origin'][0] = 101
        fhdr, message, raiser = self.log_chk(hdr, 20)
        assert fhdr == hdr
        assert message == 'very large origin values relative to dims; leaving as set, ignoring for affine'
        pytest.raises(*raiser)
        dxer = self.header_class.diagnose_binaryblock
        assert dxer(hdr.binaryblock) == 'very large origin values relative to dims'