import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy.signal import convolve2d, convolve
from skimage import restoration, util
from skimage._shared import filters
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.data import astronaut, camera
from skimage.restoration import uft
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_unsupervised_wiener(dtype):
    psf = np.ones((5, 5), dtype=dtype) / 25
    data = convolve2d(test_img, psf, 'same')
    seed = 16829302
    rng = np.random.RandomState(seed)
    data += 0.1 * data.std() * rng.standard_normal(data.shape)
    data = data.astype(dtype, copy=False)
    deconvolved, _ = restoration.unsupervised_wiener(data, psf, rng=seed)
    restoration.unsupervised_wiener(data, psf, rng=seed)
    float_type = _supported_float_type(dtype)
    assert deconvolved.dtype == float_type
    rtol, atol = _get_rtol_atol(dtype)
    path = fetch('restoration/tests/camera_unsup.npy')
    np.testing.assert_allclose(deconvolved, np.load(path), rtol=rtol, atol=atol)
    _, laplacian = uft.laplacian(2, data.shape)
    otf = uft.ir2tf(psf, data.shape, is_real=False)
    assert otf.real.dtype == _supported_float_type(dtype)
    deconvolved2 = restoration.unsupervised_wiener(data, otf, reg=laplacian, is_real=False, user_params={'callback': lambda x: None, 'max_num_iter': 200, 'min_num_iter': 30}, rng=seed)[0]
    assert deconvolved2.real.dtype == float_type
    path = fetch('restoration/tests/camera_unsup2.npy')
    np.testing.assert_allclose(np.real(deconvolved2), np.load(path), rtol=rtol, atol=atol)