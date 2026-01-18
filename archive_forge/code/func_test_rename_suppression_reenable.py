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
def test_rename_suppression_reenable(self):
    renamed = PointerXY(x=0, y=0).rename(x=None)
    self.assertEqual(renamed.contents, {'y': 0})
    reenabled = renamed.rename(x='foo')
    self.assertEqual(reenabled.contents, {'foo': 0, 'y': 0})