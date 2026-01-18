import numpy as np
from skimage._shared.testing import assert_almost_equal, assert_equal
from skimage import data, img_as_float
from skimage.morphology import diamond
from skimage.feature import match_template, peak_local_max
from skimage._shared import testing
def test_padding_reflect():
    template = diamond(2)
    image = np.zeros((10, 10))
    image[2:7, :3] = template[:, -3:]
    result = match_template(image, template, pad_input=True, mode='reflect')
    assert_equal(np.unravel_index(result.argmax(), result.shape), (4, 0))