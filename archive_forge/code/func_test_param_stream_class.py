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
def test_param_stream_class(self):
    stream = Params(self.inner)
    self.assertEqual(set(stream.parameters), {self.inner.param.x, self.inner.param.y})
    self.assertEqual(stream.contents, {'x': 0, 'y': 0})
    values = []

    def subscriber(**kwargs):
        values.append(kwargs)
    stream.add_subscriber(subscriber)
    self.inner.x = 1
    self.assertEqual(values, [{'x': 1, 'y': 0}])