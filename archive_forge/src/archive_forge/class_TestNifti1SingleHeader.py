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
class TestNifti1SingleHeader(TestNifti1PairHeader):
    header_class = Nifti1Header

    def test_empty(self):
        tana.TestAnalyzeHeader.test_empty(self)
        hdr = self.header_class()
        assert hdr['magic'] == hdr.single_magic
        assert hdr['scl_slope'] == 1
        assert hdr['vox_offset'] == 0

    def test_binblock_is_file(self):
        hdr = self.header_class()
        str_io = BytesIO()
        hdr.write_to(str_io)
        assert str_io.getvalue() == hdr.binaryblock + b'\x00' * 4

    def test_float128(self):
        hdr = self.header_class()
        ld_dt = np.dtype(np.longdouble)
        if have_binary128() or ld_dt == np.dtype(np.float64):
            hdr.set_data_dtype(np.longdouble)
            assert hdr.get_data_dtype() == ld_dt
        else:
            with pytest.raises(HeaderDataError):
                hdr.set_data_dtype(np.longdouble)