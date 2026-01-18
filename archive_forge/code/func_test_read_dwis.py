from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nibabel.optpkg import optional_package
from .. import dicomreaders as didr
from .test_dicomwrappers import DATA, EXPECTED_AFFINE, EXPECTED_PARAMS, IO_DATA_PATH
def test_read_dwis():
    data, aff, bs, gs = didr.read_mosaic_dwi_dir(IO_DATA_PATH, 'siemens_dwi_*.dcm.gz')
    assert data.ndim == 4
    assert_array_almost_equal(aff, EXPECTED_AFFINE)
    assert_array_almost_equal(bs, (0, EXPECTED_PARAMS[0]))
    assert_array_almost_equal(gs, (np.zeros((3,)), EXPECTED_PARAMS[1]))
    with pytest.raises(OSError):
        didr.read_mosaic_dwi_dir('improbable')