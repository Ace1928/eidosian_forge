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
class DynamicCallableMemoize(ComparisonTestCase):

    def test_dynamic_keydim_not_memoize(self):
        dmap = DynamicMap(lambda x: Curve([(0, x)]), kdims=['x'])
        self.assertEqual(dmap[0], Curve([(0, 0)]))
        self.assertEqual(dmap[1], Curve([(0, 1)]))

    def test_dynamic_keydim_memoize(self):
        dmap = DynamicMap(lambda x: Curve([(0, x)]), kdims=['x'])
        self.assertIs(dmap[0], dmap[0])

    def test_dynamic_keydim_memoize_disable(self):
        dmap = DynamicMap(Callable(lambda x: Curve([(0, x)]), memoize=False), kdims=['x'])
        first = dmap[0]
        del dmap.data[0,]
        second = dmap[0]
        self.assertIsNot(first, second)

    def test_dynamic_callable_memoize(self):
        history = deque(maxlen=10)

        def history_callback(x):
            history.append(x)
            return Curve(list(history))
        x = PointerX()
        dmap = DynamicMap(history_callback, kdims=[], streams=[x])
        x.add_subscriber(lambda **kwargs: dmap[()])
        x.event(x=1)
        x.event(x=1)
        self.assertEqual(dmap[()], Curve([1]))
        x.event(x=2)
        x.event(x=2)
        self.assertEqual(dmap[()], Curve([1, 2]))

    def test_dynamic_callable_disable_callable_memoize(self):
        history = deque(maxlen=10)

        def history_callback(x):
            history.append(x)
            return Curve(list(history))
        x = PointerX()
        callable_obj = Callable(history_callback, memoize=False)
        dmap = DynamicMap(callable_obj, kdims=[], streams=[x])
        x.add_subscriber(lambda **kwargs: dmap[()])
        x.event(x=1)
        x.event(x=1)
        self.assertEqual(dmap[()], Curve([1, 1, 1]))
        x.event(x=2)
        x.event(x=2)
        self.assertEqual(dmap[()], Curve([1, 1, 1, 2, 2, 2]))