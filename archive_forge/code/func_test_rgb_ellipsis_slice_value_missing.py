import numpy as np
import holoviews as hv
from holoviews.element.comparison import ComparisonTestCase
def test_rgb_ellipsis_slice_value_missing(self):
    rgb = hv.RGB(np.random.rand(10, 10, 3))
    try:
        rgb[..., 'Non-existent']
    except Exception as e:
        if str(e) != repr("'Non-existent' is not an available value dimension"):
            raise AssertionError('Incorrect exception raised.')