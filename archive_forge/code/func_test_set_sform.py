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
def test_set_sform(self):
    orig_aff = np.diag([2.2, 3.3, 4.3, 1])
    img = self.image_class(np.zeros((2, 3, 4)), orig_aff)
    hdr = img.header
    new_affine = np.diag([1.1, 1.1, 1.1, 1])
    qform_affine = np.diag([1.2, 1.2, 1.2, 1])
    aff_affine = np.diag([3.3, 4.5, 6.6, 1])
    img.affine[:] = aff_affine
    assert_array_almost_equal(img.affine, aff_affine)
    assert (hdr['sform_code'], hdr['qform_code']) == (2, 0)
    img.set_sform(new_affine, 1)
    assert hdr['sform_code'] == 1
    assert_array_almost_equal(hdr.get_sform(), new_affine)
    assert_array_almost_equal(img.get_sform(), new_affine)
    saff, code = img.get_sform(coded=True)
    assert code == 1
    assert_array_almost_equal(saff, new_affine)
    assert_array_almost_equal(img.affine, hdr.get_best_affine())
    img.affine[:] = aff_affine
    img.set_sform(new_affine, 1, update_affine=False)
    assert_array_almost_equal(img.affine, aff_affine)
    assert_array_almost_equal(img.get_qform(), orig_aff)
    assert_array_almost_equal(hdr.get_zooms(), [2.2, 3.3, 4.3])
    img.set_qform(None)
    assert_array_almost_equal(hdr.get_zooms(), [2.2, 3.3, 4.3])
    img.set_qform(qform_affine, 1)
    img.set_sform(new_affine, 1)
    saff, code = img.get_sform(coded=True)
    assert code == 1
    assert_array_almost_equal(saff, new_affine)
    assert_array_almost_equal(img.affine, new_affine)
    assert_array_almost_equal(hdr.get_zooms(), [1.2, 1.2, 1.2])
    img.set_sform(None)
    assert hdr['sform_code'] == 0
    assert hdr['qform_code'] == 1
    assert_array_almost_equal(hdr.get_sform(), saff)
    assert_array_almost_equal(img.affine, qform_affine)
    assert_array_almost_equal(hdr.get_best_affine(), img.affine)
    with pytest.raises(TypeError):
        img.get_sform(strange=True)
    img = self.image_class(np.zeros((2, 3, 4)), None)
    new_affine = np.eye(4)
    img.set_sform(new_affine, 2)
    assert_array_almost_equal(img.affine, new_affine)