import numpy as np
from skimage.morphology import max_tree, area_closing, area_opening
from skimage.morphology import max_tree_local_maxima, diameter_opening
from skimage.morphology import diameter_closing
from skimage.util import invert
from skimage._shared.testing import assert_array_equal, TestCase
def _full_type_test(img, param, expected, func, param_scale=False, **keywords):
    out = func(img, param, **keywords)
    assert_array_equal(out, expected)
    for dt in [np.uint32, np.uint64]:
        img_cast = img.astype(dt)
        out = func(img_cast, param, **keywords)
        exp_cast = expected.astype(dt)
        assert_array_equal(out, exp_cast)
    data_float = img.astype(np.float64)
    data_float = data_float / 255.0
    expected_float = expected.astype(np.float64)
    expected_float = expected_float / 255.0
    if param_scale:
        param_cast = param / 255.0
    else:
        param_cast = param
    for dt in [np.float32, np.float64]:
        data_cast = data_float.astype(dt)
        out = func(data_cast, param_cast, **keywords)
        exp_cast = expected_float.astype(dt)
        error_img = 255.0 * exp_cast - 255.0 * out
        error = (error_img >= 1.0).sum()
        assert error < eps
    img_signed = img.astype(np.int16)
    img_signed = img_signed - 128
    exp_signed = expected.astype(np.int16)
    exp_signed = exp_signed - 128
    for dt in [np.int8, np.int16, np.int32, np.int64]:
        img_s = img_signed.astype(dt)
        out = func(img_s, param, **keywords)
        exp_s = exp_signed.astype(dt)
        assert_array_equal(out, exp_s)