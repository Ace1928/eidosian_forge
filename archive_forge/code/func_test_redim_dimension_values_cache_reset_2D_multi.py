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
def test_redim_dimension_values_cache_reset_2D_multi(self):
    fn = lambda i, j: Curve([i, j])
    keys = [(0, 1), (1, 0), (2, 2), (2, 5), (3, 3)]
    dmap = DynamicMap(fn, kdims=['i', 'j'])[keys]
    self.assertEqual(dmap.keys(), keys)
    redimmed = dmap.redim.values(i=[2, 10, 50], j=[5, 50, 100])
    self.assertEqual(redimmed.keys(), [(2, 5)])