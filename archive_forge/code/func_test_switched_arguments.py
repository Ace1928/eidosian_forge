import numpy as np
from skimage._shared.testing import assert_almost_equal, assert_equal
from skimage import data, img_as_float
from skimage.morphology import diamond
from skimage.feature import match_template, peak_local_max
from skimage._shared import testing
def test_switched_arguments():
    image = np.ones((5, 5))
    template = np.ones((3, 3))
    with testing.raises(ValueError):
        match_template(template, image)