import holoviews
import pytest
from holoviews.core import Store
from holoviews.element import Area, Curve
from hvplot.backend_transforms import (
@pytest.mark.parametrize(('width', 'height', 'aspect', 'opts'), ((300, 100, None, {'aspect': 3.0, 'fig_size': 100.0}), (300, None, 3, {'aspect': 3, 'fig_size': 100.0}), (None, 300, 3, {'aspect': 3, 'fig_size': 100.0}), (300, None, None, {'fig_size': 100.0}), (None, 300, None, {'fig_size': 100.0})))
def test_transform_size_to_mpl(width, height, aspect, opts):
    assert _transform_size_to_mpl(width, height, aspect) == opts