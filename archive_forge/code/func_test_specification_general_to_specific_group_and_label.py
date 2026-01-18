import os
import pickle
import numpy as np
import pytest
from holoviews import (
from holoviews.core.options import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import mpl # noqa
from holoviews.plotting import bokeh # noqa
from holoviews.plotting import plotly # noqa
def test_specification_general_to_specific_group_and_label(self):
    """
        Test order of specification starting with general and moving
        to specific
        """
    options = self.initialize_option_tree()
    obj = Image(np.random.rand(10, 10), group='SomeGroup', label='SomeLabel')
    options.Image = Options('style', cmap='viridis')
    options.Image.SomeGroup.SomeLabel = Options('style', alpha=0.2)
    expected = {'alpha': 0.2, 'cmap': 'viridis', 'interpolation': 'nearest'}
    lookup = Store.lookup_options('matplotlib', obj, 'style')
    self.assertEqual(lookup.kwargs, expected)
    node1 = options.Image.groups['style']
    node2 = options.Image.SomeGroup.SomeLabel.groups['style']
    self.assertEqual(node1.kwargs, {'cmap': 'viridis', 'interpolation': 'nearest'})
    self.assertEqual(node2.kwargs, {'alpha': 0.2})