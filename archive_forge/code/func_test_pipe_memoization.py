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
def test_pipe_memoization(self):

    def points(data):
        subscriber.call_count += 1
        return Points([(0, data)])
    stream = Pipe(data=0)
    dmap = DynamicMap(points, streams=[stream])

    def cb():
        dmap[()]
    subscriber = _TestSubscriber(cb)
    stream.add_subscriber(subscriber)
    dmap[()]
    stream.send(1)
    self.assertEqual(subscriber.call_count, 3)