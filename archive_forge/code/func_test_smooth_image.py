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
def test_smooth_image(caplog):
    data = np.arange(24, dtype='int32').reshape((2, 3, 4))
    aff = np.diag([-4, 5, 6, 1])
    img = Nifti1Image(data, aff)
    out_img = smooth_image(img, 0)
    assert_array_equal(out_img.affine, img.affine)
    assert_array_equal(out_img.shape, img.shape)
    assert_array_equal(out_img.dataobj, data)
    sd = fwhm2sigma(np.true_divide(8, [4, 5, 6]))
    exp_out = spnd.gaussian_filter(data, sd, mode='nearest')
    assert_array_equal(smooth_image(img, 8).dataobj, exp_out)
    assert_array_equal(smooth_image(img, [8, 8, 8]).dataobj, exp_out)
    with pytest.raises(ValueError):
        smooth_image(img, [8, 8])
    mixed_sd = fwhm2sigma(np.true_divide([8, 7, 6], [4, 5, 6]))
    exp_out = spnd.gaussian_filter(data, mixed_sd, mode='nearest')
    assert_array_equal(smooth_image(img, [8, 7, 6]).dataobj, exp_out)
    img_2d = Nifti1Image(data[0], aff)
    exp_out = spnd.gaussian_filter(data[0], sd[:2], mode='nearest')
    assert_array_equal(smooth_image(img_2d, 8).dataobj, exp_out)
    assert_array_equal(smooth_image(img_2d, [8, 8]).dataobj, exp_out)
    with pytest.raises(ValueError):
        smooth_image(img_2d, [8, 8, 8])
    data_4d = np.arange(24 * 5, dtype='int32').reshape((2, 3, 4, 5))
    img_4d = Nifti1Image(data_4d, aff)
    exp_out = spnd.gaussian_filter(data_4d, list(sd) + [0], mode='nearest')
    assert_array_equal(smooth_image(img_4d, 8).dataobj, exp_out)
    with pytest.raises(ValueError):
        smooth_image(img_4d, [8, 8, 8])
    exp_out = spnd.gaussian_filter(data, sd, mode='constant')
    assert_array_equal(smooth_image(img, 8, mode='constant').dataobj, exp_out)
    exp_out = spnd.gaussian_filter(data, sd, mode='constant', cval=99)
    assert_array_equal(smooth_image(img, 8, mode='constant', cval=99).dataobj, exp_out)
    img_ni1 = Nifti1Image(data, np.eye(4))
    img_ni2 = Nifti2Image(data, np.eye(4))
    with caplog.at_level(logging.CRITICAL):
        assert smooth_image(img_ni2, 0).__class__ == Nifti1Image
    with caplog.at_level(logging.CRITICAL):
        assert smooth_image(img_ni1, 0, out_class=Nifti2Image).__class__ == Nifti2Image
    assert smooth_image(img_ni2, 0, out_class=None).__class__ == Nifti2Image