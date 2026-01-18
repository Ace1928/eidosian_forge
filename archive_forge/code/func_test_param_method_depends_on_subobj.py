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
def test_param_method_depends_on_subobj(self):
    inner = self.innersubobj(sub=self.inner())
    stream = ParamMethod(inner.subobj_method)
    self.assertEqual(set(stream.parameters), {inner.sub.param.x})
    values = []

    def subscriber(**kwargs):
        values.append(kwargs)
    stream.add_subscriber(subscriber)
    inner.sub.x = 2
    self.assertEqual(values, [{}])