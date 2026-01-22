import io
from os.path import dirname
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from packaging.version import Version
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.cifti2.parse_cifti2 import _Cifti2AsNiftiHeader
from nibabel.tests import test_nifti2 as tn2
from nibabel.tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from nibabel.tmpdirs import InTemporaryDirectory
class TestCifti2SingleHeader(tn2.TestNifti2SingleHeader):
    header_class = _Cifti2AsNiftiHeader
    _pixdim_message = 'pixdim[1,2,3] should be zero or positive'

    def test_pixdim_checks(self):
        hdr_t = self.header_class()
        for i in (1, 2, 3):
            hdr = hdr_t.copy()
            hdr['pixdim'][i] = -1
            assert self._dxer(hdr) == self._pixdim_message

    def test_nifti_qfac_checks(self):
        hdr = self.header_class()
        hdr['pixdim'][0] = 1
        self.log_chk(hdr, 0)
        hdr['pixdim'][0] = 0
        self.log_chk(hdr, 0)
        hdr['pixdim'][0] = -1
        self.log_chk(hdr, 0)
        hdr['pixdim'][0] = 2
        fhdr, message, raiser = self.log_chk(hdr, 20)
        assert fhdr['pixdim'][0] == 1
        assert message == 'pixdim[0] (qfac) should be 1 (default) or 0 or -1; setting qfac to 1'

    def test_pixdim_log_checks(self):
        HC = self.header_class
        hdr = HC()
        hdr['pixdim'][1] = -2
        fhdr, message, raiser = self.log_chk(hdr, 35)
        assert fhdr['pixdim'][1] == 2
        assert message == self._pixdim_message + '; setting to abs of pixdim values'
        pytest.raises(*raiser)
        hdr = HC()
        hdr['pixdim'][1:4] = 0
        fhdr, message, raiser = self.log_chk(hdr, 0)
        assert raiser == ()