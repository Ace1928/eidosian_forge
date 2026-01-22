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
class TestBufferDataFrameStream(ComparisonTestCase):

    def test_init_buffer_dframe(self):
        data = pd.DataFrame({'x': np.array([1]), 'y': np.array([2])})
        buff = Buffer(data, index=False)
        self.assertEqual(buff.data, data)

    def test_init_buffer_dframe_with_index(self):
        data = pd.DataFrame({'x': np.array([1]), 'y': np.array([2])})
        buff = Buffer(data)
        self.assertEqual(buff.data, data.reset_index())

    def test_buffer_dframe_send(self):
        data = pd.DataFrame({'x': np.array([0]), 'y': np.array([1])})
        buff = Buffer(data, index=False)
        buff.send(pd.DataFrame({'x': np.array([1]), 'y': np.array([2])}))
        dframe = pd.DataFrame({'x': np.array([0, 1]), 'y': np.array([1, 2])})
        self.assertEqual(buff.data.values, dframe.values)

    def test_buffer_dframe_send_with_index(self):
        data = pd.DataFrame({'x': np.array([0]), 'y': np.array([1])})
        buff = Buffer(data)
        buff.send(pd.DataFrame({'x': np.array([1]), 'y': np.array([2])}))
        dframe = pd.DataFrame({'x': np.array([0, 1]), 'y': np.array([1, 2])}, index=[0, 0])
        self.assertEqual(buff.data.values, dframe.reset_index().values)

    def test_buffer_dframe_larger_than_length(self):
        data = pd.DataFrame({'x': np.array([0]), 'y': np.array([1])})
        buff = Buffer(data, length=1, index=False)
        chunk = pd.DataFrame({'x': np.array([1]), 'y': np.array([2])})
        buff.send(chunk)
        self.assertEqual(buff.data.values, chunk.values)

    def test_buffer_dframe_patch_larger_than_length(self):
        data = pd.DataFrame({'x': np.array([0]), 'y': np.array([1])})
        buff = Buffer(data, length=1, index=False)
        chunk = pd.DataFrame({'x': np.array([1, 2]), 'y': np.array([2, 3])})
        buff.send(chunk)
        dframe = pd.DataFrame({'x': np.array([2]), 'y': np.array([3])})
        self.assertEqual(buff.data.values, dframe.values)

    def test_buffer_dframe_send_verify_column_fail(self):
        data = pd.DataFrame({'x': np.array([0]), 'y': np.array([1])})
        buff = Buffer(data, index=False)
        error = "Input expected to have columns \\['x', 'y'\\], got \\['x'\\]"
        with self.assertRaisesRegex(IndexError, error):
            buff.send(pd.DataFrame({'x': np.array([2])}))

    def test_clear_buffer_dframe_with_index(self):
        data = pd.DataFrame({'a': [1, 2, 3]})
        buff = Buffer(data)
        buff.clear()
        self.assertEqual(buff.data, data.iloc[:0, :].reset_index())