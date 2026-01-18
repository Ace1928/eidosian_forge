import numpy as np
import holoviews as hv
from holoviews.element.comparison import ComparisonTestCase
def test_scatter_ellipsis_value_missing(self):
    try:
        hv.Scatter(range(10))[..., 'Non-existent']
    except Exception as e:
        if str(e) != "'Non-existent' is not an available value dimension":
            raise AssertionError('Incorrect exception raised.')