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
def test_selection_expr_stream_invert_xaxis_yaxis_2D_elements(self):
    element_type_2D = [Points]
    for element_type in element_type_2D:
        element = element_type(([1, 2, 3], [1, 5, 10])).opts(invert_xaxis=True, invert_yaxis=True)
        expr_stream = SelectionExpr(element)
        self.assertEqual(len(expr_stream.input_streams), 3)
        self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
        self.assertIsInstance(expr_stream.input_streams[1], Lasso)
        self.assertIsInstance(expr_stream.input_streams[2], Selection1D)
        self.assertIsNone(expr_stream.bbox)
        self.assertIsNone(expr_stream.selection_expr)
        expr_stream.input_streams[0].event(bounds=(3, 4, 1, 1))
        self.assertEqual(repr(expr_stream.selection_expr), repr((dim('x') >= 1) & (dim('x') <= 3) & ((dim('y') >= 1) & (dim('y') <= 4))))
        self.assertEqual(expr_stream.bbox, {'x': (1, 3), 'y': (1, 4)})