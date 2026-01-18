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
def test_param_parameter_instance_separate_parameters(self):
    inner = self.inner()
    valid, invalid = Stream._process_streams([inner.param.x, inner.param.y])
    xparam, yparam = valid
    self.assertIs(xparam.parameterized, inner)
    self.assertEqual(xparam.parameters, [inner.param.x])
    self.assertIs(yparam.parameterized, inner)
    self.assertEqual(yparam.parameters, [inner.param.y])