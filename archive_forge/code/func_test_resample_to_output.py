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
def test_resample_to_output(caplog):
    data = np.arange(24, dtype='int32').reshape((2, 3, 4))
    img = Nifti1Image(data, np.eye(4))
    img2 = resample_to_output(img)
    assert_array_equal(img2.shape, (2, 3, 4))
    assert_array_equal(img2.affine, np.eye(4))
    assert_array_equal(img2.dataobj, data)
    for vox_sizes in (None, 1, [1, 1, 1]):
        img2 = resample_to_output(img, vox_sizes)
        assert_array_equal(img2.shape, (2, 3, 4))
        assert_array_equal(img2.affine, np.eye(4))
        assert_array_equal(img2.dataobj, data)
    img2 = resample_to_output(img, vox_sizes)
    img_2d = Nifti1Image(data[0], np.eye(4))
    for vox_sizes in (None, 1, (1, 1), (1, 1, 1)):
        img3 = resample_to_output(img_2d, vox_sizes)
        assert_array_equal(img3.shape, (3, 4, 1))
        assert_array_equal(img3.affine, np.eye(4))
        assert_array_equal(img3.dataobj, data[0][..., None])
    img_1d = Nifti1Image(data[0, 0], np.eye(4))
    img3 = resample_to_output(img_1d)
    assert_array_equal(img3.shape, (4, 1, 1))
    assert_array_equal(img3.affine, np.eye(4))
    assert_array_equal(img3.dataobj, data[0, 0][..., None, None])
    img_4d = Nifti1Image(data.reshape(2, 3, 2, 2), np.eye(4))
    with pytest.raises(ValueError):
        resample_to_output(img_4d)
    for in_shape, in_aff, vox, out_shape, out_aff in get_outspace_params():
        in_n_dim = len(in_shape)
        if in_n_dim < 3:
            in_shape = in_shape + (1,) * (3 - in_n_dim)
            if not vox is None:
                vox = vox + (1,) * (3 - in_n_dim)
            assert len(out_shape) == in_n_dim
            out_shape = out_shape + (1,) * (3 - in_n_dim)
        img = Nifti1Image(np.ones(in_shape), in_aff)
        out_img = resample_to_output(img, vox)
        assert_all_in(in_shape, in_aff, out_img.shape, out_img.affine)
        assert out_img.shape == out_shape
        assert_almost_equal(out_img.affine, out_aff)
    out_img = resample_to_output(Nifti1Image(data, np.diag([-1, 1, 1, 1])))
    assert_array_equal(out_img.dataobj, np.flipud(data))
    out_img = resample_to_output(Nifti1Image(data, np.diag([4, 5, 6, 1])))
    with pytest.warns(UserWarning):
        exp_out = spnd.affine_transform(data, [1 / 4, 1 / 5, 1 / 6], output_shape=(5, 11, 19))
    assert_array_equal(out_img.dataobj, exp_out)
    out_img = resample_to_output(Nifti1Image(data, np.diag([4, 5, 6, 1])), [4, 5, 6])
    assert_array_equal(out_img.dataobj, data)
    rot_3 = from_matvec(euler2mat(np.pi / 4), [0, 0, 0])
    rot_3_img = Nifti1Image(data, rot_3)
    out_img = resample_to_output(rot_3_img)
    exp_shape = (4, 4, 4)
    assert out_img.shape == exp_shape
    exp_aff = np.array([[1, 0, 0, -2 * np.cos(np.pi / 4)], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert_almost_equal(out_img.affine, exp_aff)
    rzs, trans = to_matvec(np.dot(npl.inv(rot_3), exp_aff))
    exp_out = spnd.affine_transform(data, rzs, trans, exp_shape)
    assert_almost_equal(out_img.dataobj, exp_out)
    assert_almost_equal(resample_to_output(rot_3_img, order=0).dataobj, spnd.affine_transform(data, rzs, trans, exp_shape, order=0))
    assert_almost_equal(resample_to_output(rot_3_img, cval=99).dataobj, spnd.affine_transform(data, rzs, trans, exp_shape, cval=99))
    assert_almost_equal(resample_to_output(rot_3_img, mode='nearest').dataobj, spnd.affine_transform(data, rzs, trans, exp_shape, mode='nearest'))
    img_ni1 = Nifti2Image(data, np.eye(4))
    img_ni2 = Nifti2Image(data, np.eye(4))
    with caplog.at_level(logging.CRITICAL):
        assert resample_to_output(img_ni2).__class__ == Nifti1Image
    with caplog.at_level(logging.CRITICAL):
        assert resample_to_output(img_ni1, out_class=Nifti2Image).__class__ == Nifti2Image
    assert resample_to_output(img_ni2, out_class=None).__class__ == Nifti2Image