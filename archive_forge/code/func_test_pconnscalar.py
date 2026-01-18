import numpy as np
import pytest
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory
from ...testing import (
def test_pconnscalar():
    parcel_map = create_parcel_map((0, 1))
    scalar_map = create_scalar_map((2,))
    matrix = ci.Cifti2Matrix()
    matrix.extend((parcel_map, scalar_map))
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(4, 4, 2)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_PARCELLATED_PARCELLATED_SCALAR')
    with InTemporaryDirectory():
        ci.save(img, 'test.pconnscalar.nii')
        img2 = ci.load('test.pconnscalar.nii')
        assert img.nifti_header.get_intent()[0] == 'ConnPPSc'
        assert isinstance(img2, ci.Cifti2Image)
        assert_array_equal(img2.get_fdata(), data)
        assert img2.header.matrix.get_index_map(0) == img2.header.matrix.get_index_map(1)
        check_parcel_map(img2.header.matrix.get_index_map(0))
        check_scalar_map(img2.header.matrix.get_index_map(2))
        del img2