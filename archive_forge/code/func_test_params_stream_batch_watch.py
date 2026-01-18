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
def test_params_stream_batch_watch(self):
    tap = Tap(x=0, y=1)
    params = Params(parameters=[tap.param.x, tap.param.y])
    values = []

    def subscriber(**kwargs):
        values.append(kwargs)
    params.add_subscriber(subscriber)
    tap.param.trigger('x', 'y')
    assert values == [{'x': 0, 'y': 1}]
    tap.event(x=1, y=2)
    assert values == [{'x': 0, 'y': 1}, {'x': 1, 'y': 2}]