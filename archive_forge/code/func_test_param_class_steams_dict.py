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
def test_param_class_steams_dict(self):

    class ClassParamExample(param.Parameterized):
        x = param.Number(default=1)

    def test(x):
        return Points([x])
    dmap = DynamicMap(test, streams=dict(x=ClassParamExample.param.x))
    ClassParamExample.x = 10
    self.assertEqual(dmap[()], Points([10]))