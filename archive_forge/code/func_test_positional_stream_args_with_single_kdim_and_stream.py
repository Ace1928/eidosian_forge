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
def test_positional_stream_args_with_single_kdim_and_stream(self):
    fn = lambda i, s: Points([i, i]) + Curve([s['x'], s['y']])
    xy_stream = XY(x=1, y=2)
    dmap = DynamicMap(fn, kdims=['i'], streams=[xy_stream], positional_stream_args=True)
    self.assertEqual(dmap[6], Points([6, 6]) + Curve([1, 2]))
    xy_stream.event(x=5, y=7)
    self.assertEqual(dmap[3], Points([3, 3]) + Curve([5, 7]))