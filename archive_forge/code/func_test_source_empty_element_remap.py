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
def test_source_empty_element_remap(self):
    points = Points([])
    stream = PointerX(source=points)
    self.assertIs(stream.source, points)
    curve = Curve([])
    stream.source = curve
    self.assertNotIn(points, Stream.registry)
    self.assertIn(curve, Stream.registry)