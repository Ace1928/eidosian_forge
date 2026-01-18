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
def test_param_stream_instance_overlapping_parameters(self):
    inner = self.inner()
    params1 = Params(inner)
    params2 = Params(inner)
    Stream._process_streams([params1, params2])
    self.log_handler.assertContains('WARNING', "['x', 'y']")