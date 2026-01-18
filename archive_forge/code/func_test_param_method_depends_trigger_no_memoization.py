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
def test_param_method_depends_trigger_no_memoization(self):
    inner = self.inner()
    stream = ParamMethod(inner.method)
    self.assertEqual(set(stream.parameters), {inner.param.x})
    self.assertEqual(stream.contents, {})
    values = []

    def subscriber(**kwargs):
        values.append(kwargs)
    stream.add_subscriber(subscriber)
    inner.x = 2
    inner.param.trigger('x')
    self.assertEqual(values, [{}, {}])