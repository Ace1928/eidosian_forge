import numpy as np
import holoviews as hv
from holoviews.element.comparison import ComparisonTestCase
def test_raster_ellipsis_slice_value_missing(self):
    data = np.random.rand(10, 10)
    try:
        hv.Raster(data)[..., 'Non-existent']
    except Exception as e:
        if "'z' is the only selectable value dimension" not in str(e):
            raise AssertionError('Unexpected exception.')