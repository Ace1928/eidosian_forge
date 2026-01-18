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
def test_periodic_counter_blocking(self):

    class Counter:

        def __init__(self):
            self.count = 0

        def __call__(self):
            self.count += 1
            return Curve([1, 2, 3])
    next_stream = Stream.define('Next')()
    counter = Counter()
    dmap = DynamicMap(counter, streams=[next_stream])
    next_stream.add_subscriber(lambda **kwargs: dmap[()])
    dmap.periodic(0.01, 100)
    self.assertEqual(counter.count, 100)