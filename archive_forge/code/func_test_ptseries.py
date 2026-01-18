import numpy as np
import pytest
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory
from ...testing import (
def test_ptseries():
    series_map = create_series_map((0,))
    parcel_map = create_parcel_map((1,))
    matrix = ci.Cifti2Matrix()
    matrix.extend((series_map, parcel_map))
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(13, 4)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_PARCELLATED_SERIES')
    with InTemporaryDirectory():
        ci.save(img, 'test.ptseries.nii')
        img2 = nib.load('test.ptseries.nii')
        assert img2.nifti_header.get_intent()[0] == 'ConnParcelSries'
        assert isinstance(img2, ci.Cifti2Image)
        assert_array_equal(img2.get_fdata(), data)
        check_series_map(img2.header.matrix.get_index_map(0))
        check_parcel_map(img2.header.matrix.get_index_map(1))
        del img2