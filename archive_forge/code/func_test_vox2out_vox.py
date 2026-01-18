import numpy as np
import numpy.linalg as npl
import pytest
from numpy.testing import assert_almost_equal
from ..affines import apply_affine, from_matvec
from ..eulerangles import euler2mat
from ..nifti1 import Nifti1Image
from ..spaces import slice2volume, vox2out_vox
def test_vox2out_vox():
    shape, aff = vox2out_vox(((2, 3, 4), np.eye(4)))
    assert shape == (2, 3, 4)
    assert (aff == np.eye(4)).all()
    for in_shape, in_aff, vox, out_shape, out_aff in get_outspace_params():
        img = Nifti1Image(np.ones(in_shape), in_aff)
        for input in ((in_shape, in_aff), img):
            shape, aff = vox2out_vox(input, vox)
            assert_all_in(in_shape, in_aff, shape, aff)
            assert shape == out_shape
            assert_almost_equal(aff, out_aff)
            assert isinstance(shape, tuple)
            assert isinstance(shape[0], int)
    with pytest.raises(ValueError):
        vox2out_vox(((2, 3, 4, 5), np.eye(4)))
    with pytest.raises(ValueError):
        vox2out_vox(((2, 3, 4, 5, 6), np.eye(4)))
    with pytest.raises(ValueError):
        vox2out_vox(((2, 3, 4), np.eye(4), [-1, 1, 1]))
    with pytest.raises(ValueError):
        vox2out_vox(((2, 3, 4), np.eye(4), [1, 0, 1]))