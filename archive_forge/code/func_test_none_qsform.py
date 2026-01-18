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
def test_none_qsform(self):
    img_klass = self.image_class
    hdr_klass = img_klass.header_class
    shape = (2, 3, 4)
    data = np.arange(24, dtype='f4').reshape((2, 3, 4))
    aff = from_matvec(euler2mat(0.1, 0.2, 0.3), [11, 12, 13])
    for hdr in (None, hdr_klass()):
        img = img_klass(data, aff, hdr)
        assert_almost_equal(img.affine, aff)
        assert_almost_equal(img.header.get_sform(), aff)
        assert_almost_equal(img.header.get_qform(), aff)
    hdr = hdr_klass()
    hdr.set_data_shape(shape)
    default_aff = hdr.get_best_affine()
    img = img_klass(data, default_aff, None)
    assert_almost_equal(img.header.get_sform(), default_aff)
    assert_almost_equal(img.header.get_qform(), default_aff)
    img = img_klass(data, None, None)
    assert_almost_equal(img.header.get_sform(), np.diag([0, 0, 0, 1]))
    assert_almost_equal(img.header.get_qform(), np.eye(4))