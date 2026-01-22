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
class TestPipeStream(ComparisonTestCase):

    def test_pipe_send(self):

        def subscriber(data):
            subscriber.test = data
        subscriber.test = None
        pipe = Pipe()
        pipe.add_subscriber(subscriber)
        pipe.send('Test')
        self.assertEqual(pipe.data, 'Test')
        self.assertEqual(subscriber.test, 'Test')

    def test_pipe_event(self):

        def subscriber(data):
            subscriber.test = data
        subscriber.test = None
        pipe = Pipe()
        pipe.add_subscriber(subscriber)
        pipe.event(data='Test')
        self.assertEqual(pipe.data, 'Test')
        self.assertEqual(subscriber.test, 'Test')

    def test_pipe_update(self):
        pipe = Pipe()
        pipe.event(data='Test')
        self.assertEqual(pipe.data, 'Test')