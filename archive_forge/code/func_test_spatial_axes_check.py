import logging
from os.path import dirname
from os.path import join as pjoin
import numpy as np
import numpy.linalg as npl
from nibabel.optpkg import optional_package
import unittest
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
import nibabel as nib
from nibabel.affines import AffineError, apply_affine, from_matvec, to_matvec, voxel_sizes
from nibabel.eulerangles import euler2mat
from nibabel.nifti1 import Nifti1Image
from nibabel.nifti2 import Nifti2Image
from nibabel.orientations import aff2axcodes, inv_ornt_aff
from nibabel.processing import (
from nibabel.testing import assert_allclose_safely
from nibabel.tests.test_spaces import assert_all_in, get_outspace_params
from .test_imageclasses import MINC_3DS, MINC_4DS
@needs_scipy
def test_spatial_axes_check(caplog):
    for fname in MINC_3DS + OTHER_IMGS:
        img = nib.load(pjoin(DATA_DIR, fname))
        with caplog.at_level(logging.CRITICAL):
            s_img = smooth_image(img, 0)
        assert_array_equal(img.dataobj, s_img.dataobj)
        with caplog.at_level(logging.CRITICAL):
            out = resample_from_to(img, img, mode='nearest')
        assert_almost_equal(img.dataobj, out.dataobj)
        if len(img.shape) > 3:
            continue
        out = resample_to_output(img, voxel_sizes(img.affine))
    for fname in MINC_4DS:
        img = nib.load(pjoin(DATA_DIR, fname))
        with pytest.raises(ValueError):
            smooth_image(img, 0)
        with pytest.raises(ValueError):
            resample_from_to(img, img, mode='nearest')
        with pytest.raises(ValueError):
            resample_to_output(img, voxel_sizes(img.affine))