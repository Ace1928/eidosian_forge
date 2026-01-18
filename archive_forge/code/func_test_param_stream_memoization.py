from collections import defaultdict
from unittest import SkipTest
import pandas as pd
import param
import pytest
from panel.widgets import IntSlider
import holoviews as hv
from holoviews.core.spaces import DynamicMap
from holoviews.core.util import Version
from holoviews.element import Curve, Histogram, Points, Polygons, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import *  # noqa (Test all available streams)
from holoviews.util import Dynamic, extension
from holoviews.util.transform import dim
from .utils import LoggingComparisonTestCase
def test_param_stream_memoization(self):
    inner = self.inner_action()
    stream = Params(inner, ['action', 'x'])
    self.assertEqual(set(stream.parameters), {inner.param.action, inner.param.x})
    values = []

    def subscriber(**kwargs):
        values.append(kwargs)
        self.assertEqual(set(stream.hashkey), {f'{id(inner)} action', f'{id(inner)} x', '_memoize_key'})
    stream.add_subscriber(subscriber)
    inner.action(inner)
    inner.x = 0
    self.assertEqual(values, [{'action': inner.action, 'x': 0}])