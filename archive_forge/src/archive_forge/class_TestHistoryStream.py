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
class TestHistoryStream(ComparisonTestCase):

    def test_initial_history_stream_values(self):
        val = Val(v=1.0)
        history = History(val)
        self.assertEqual(history.contents, {'values': [val.contents]})

    def test_history_stream_values_appended(self):
        val = Val(v=1.0)
        history = History(val)
        val.event(v=2.0)
        val.event(v=3.0)
        self.assertEqual(history.contents, {'values': [{'v': 1.0}, {'v': 2.0}, {'v': 3.0}]})
        history.clear_history()
        self.assertEqual(history.contents, {'values': []})

    def test_history_stream_trigger_callbacks(self):
        val = Val(v=1.0)
        history = History(val)
        callback_input = []

        def cb(**kwargs):
            callback_input.append(kwargs)
        history.add_subscriber(cb)
        self.assertEqual(callback_input, [])
        del callback_input[:]
        val.event(v=2.0)
        self.assertEqual(callback_input[0], {'values': [{'v': 1.0}, {'v': 2.0}]})
        del callback_input[:]
        val.event(v=3.0)
        self.assertEqual(callback_input[0], {'values': [{'v': 1.0}, {'v': 2.0}, {'v': 3.0}]})
        del callback_input[:]
        history.clear_history()
        history.event()
        self.assertEqual(callback_input[0], {'values': []})
        del callback_input[:]
        val.event(v=4.0)
        self.assertEqual(callback_input[0], {'values': [{'v': 4.0}]})