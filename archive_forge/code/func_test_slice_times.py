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
def test_slice_times(self):
    hdr = self.header_class()
    with pytest.raises(HeaderDataError):
        hdr.get_slice_times()
    hdr.set_dim_info(slice=2)
    with pytest.raises(HeaderDataError):
        hdr.get_slice_times()
    hdr.set_data_shape((1, 1, 7))
    with pytest.raises(HeaderDataError):
        hdr.get_slice_times()
    hdr.set_slice_duration(0.1)
    _stringer = lambda val: val is not None and '%2.1f' % val or None
    _print_me = lambda s: list(map(_stringer, s))
    hdr['slice_code'] = slice_order_codes['sequential increasing']
    assert _print_me(hdr.get_slice_times()) == ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6']
    hdr['slice_start'] = 1
    hdr['slice_end'] = 5
    assert _print_me(hdr.get_slice_times()) == [None, '0.0', '0.1', '0.2', '0.3', '0.4', None]
    hdr['slice_code'] = slice_order_codes['sequential decreasing']
    assert _print_me(hdr.get_slice_times()) == [None, '0.4', '0.3', '0.2', '0.1', '0.0', None]
    hdr['slice_code'] = slice_order_codes['alternating increasing']
    assert _print_me(hdr.get_slice_times()) == [None, '0.0', '0.3', '0.1', '0.4', '0.2', None]
    hdr['slice_code'] = slice_order_codes['alternating decreasing']
    assert _print_me(hdr.get_slice_times()) == [None, '0.2', '0.4', '0.1', '0.3', '0.0', None]
    hdr['slice_code'] = slice_order_codes['alternating increasing 2']
    assert _print_me(hdr.get_slice_times()) == [None, '0.2', '0.0', '0.3', '0.1', '0.4', None]
    hdr['slice_code'] = slice_order_codes['alternating decreasing 2']
    assert _print_me(hdr.get_slice_times()) == [None, '0.4', '0.1', '0.3', '0.0', '0.2', None]
    hdr = self.header_class()
    hdr.set_dim_info(slice=2)
    times = [None, 0.2, 0.4, 0.1, 0.3, 0.0, None]
    with pytest.raises(HeaderDataError):
        hdr.set_slice_times(times)
    hdr.set_data_shape([1, 1, 7])
    with pytest.raises(HeaderDataError):
        hdr.set_slice_times(times[:-1])
    with pytest.raises(HeaderDataError):
        hdr.set_slice_times((None,) * len(times))
    n_mid_times = times[:]
    n_mid_times[3] = None
    with pytest.raises(HeaderDataError):
        hdr.set_slice_times(n_mid_times)
    funny_times = times[:]
    funny_times[3] = 0.05
    with pytest.raises(HeaderDataError):
        hdr.set_slice_times(funny_times)
    hdr.set_slice_times(times)
    assert hdr.get_value_label('slice_code') == 'alternating decreasing'
    assert hdr['slice_start'] == 1
    assert hdr['slice_end'] == 5
    assert_array_almost_equal(hdr['slice_duration'], 0.1)
    hdr2 = self.header_class()
    hdr2.set_dim_info(slice=2)
    hdr2.set_slice_duration(0.1)
    hdr2.set_data_shape((1, 1, 2))
    with pytest.warns(UserWarning) as w:
        hdr2.set_slice_times([0.1, 0])
        assert len(w) == 1
    assert hdr2.get_value_label('slice_code') == 'sequential decreasing'
    with pytest.warns(UserWarning) as w:
        hdr2.set_slice_times([0, 0.1])
        assert len(w) == 1
    assert hdr2.get_value_label('slice_code') == 'sequential increasing'