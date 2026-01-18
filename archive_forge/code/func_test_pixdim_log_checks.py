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