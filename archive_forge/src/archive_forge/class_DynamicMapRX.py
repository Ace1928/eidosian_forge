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
class DynamicMapRX(ComparisonTestCase):

    def test_dynamic_rx(self):
        freq = param.rx(1)
        rx_curve = param.rx(sine_array)(0, freq).rx.pipe(Curve)
        dmap = DynamicMap(rx_curve)
        assert len(dmap.streams) == 1
        pstream = dmap.streams[0]
        assert isinstance(pstream, Params)
        assert len(pstream.parameters) == 2
        fn_param, freq_param = pstream.parameters
        assert getattr(fn_param.owner, fn_param.name) == sine_array
        assert getattr(freq_param.owner, freq_param.name) == 1
        self.assertEqual(dmap[()], Curve(sine_array(0, 1)))
        freq.rx.value = 2
        self.assertEqual(dmap[()], Curve(sine_array(0, 2)))