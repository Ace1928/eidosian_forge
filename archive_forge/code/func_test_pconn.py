import numpy as np
import pytest
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory
from ...testing import (
def test_pconn():
    mapping = create_parcel_map((0, 1))
    matrix = ci.Cifti2Matrix()
    matrix.append(mapping)
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(4, 4)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_PARCELLATED')
    with InTemporaryDirectory():
        ci.save(img, 'test.pconn.nii')
        img2 = ci.load('test.pconn.nii')
        assert img.nifti_header.get_intent()[0] == 'ConnParcels'
        assert isinstance(img2, ci.Cifti2Image)
        assert_array_equal(img2.get_fdata(), data)
        assert img2.header.matrix.get_index_map(0) == img2.header.matrix.get_index_map(1)
        check_parcel_map(img2.header.matrix.get_index_map(0))
        del img2