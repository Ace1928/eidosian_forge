import time
import uuid
from collections import deque
import numpy as np
import param
import pytest
from holoviews import Dimension, GridSpace, Layout, NdLayout, NdOverlay
from holoviews.core.options import Store
from holoviews.core.spaces import Callable, DynamicMap, HoloMap
from holoviews.element import Curve, Image, Points, Scatter, Text
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import histogram
from holoviews.plotting.util import initialize_dynamic
from holoviews.streams import (
from holoviews.util import Dynamic
from ..utils import LoggingComparisonTestCase
from .test_dimensioned import CustomBackendTestCase, ExampleElement
def test_redim_dimension_values_cache_reset_1D(self):
    fn = lambda i: Curve([i, i])
    dmap = DynamicMap(fn, kdims=['i'])[{0, 1, 2, 3, 4, 5}]
    self.assertEqual(dmap.keys(), [0, 1, 2, 3, 4, 5])
    redimmed = dmap.redim.values(i=[2, 3, 5, 6, 8])
    self.assertEqual(redimmed.keys(), [2, 3, 5])