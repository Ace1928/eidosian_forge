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
class DynamicTransferStreams(ComparisonTestCase):

    def setUp(self):
        self.dimstream = PointerX(x=0)
        self.stream = PointerY(y=0)
        self.dmap = DynamicMap(lambda x, y, z: Curve([x, y, z]), kdims=['x', 'z'], streams=[self.stream, self.dimstream])

    def test_dynamic_redim_inherits_streams(self):
        redimmed = self.dmap.redim.range(z=(0, 5))
        self.assertEqual(redimmed.streams, self.dmap.streams)

    def test_dynamic_relabel_inherits_streams(self):
        relabelled = self.dmap.relabel(label='Test')
        self.assertEqual(relabelled.streams, self.dmap.streams)

    def test_dynamic_map_inherits_streams(self):
        mapped = self.dmap.map(lambda x: x, Curve)
        self.assertEqual(mapped.streams, self.dmap.streams)

    def test_dynamic_select_inherits_streams(self):
        selected = self.dmap.select(Curve, x=(0, 5))
        self.assertEqual(selected.streams, self.dmap.streams)

    def test_dynamic_hist_inherits_streams(self):
        hist = self.dmap.hist(adjoin=False)
        self.assertEqual(hist.streams, self.dmap.streams)

    def test_dynamic_mul_inherits_dim_streams(self):
        hist = self.dmap * self.dmap
        self.assertEqual(hist.streams, self.dmap.streams[1:])

    def test_dynamic_util_inherits_dim_streams(self):
        hist = Dynamic(self.dmap)
        self.assertEqual(hist.streams, self.dmap.streams[1:])

    def test_dynamic_util_parameterized_method(self):

        class Test(param.Parameterized):
            label = param.String(default='test')

            @param.depends('label')
            def apply_label(self, obj):
                return obj.relabel(self.label)
        test = Test()
        dmap = Dynamic(self.dmap, operation=test.apply_label)
        test.label = 'custom label'
        self.assertEqual(dmap[0, 3].label, 'custom label')

    def test_dynamic_util_inherits_dim_streams_clash(self):
        exception = "The supplied stream objects PointerX\\(x=None\\) and PointerX\\(x=0\\) clash on the following parameters: \\['x'\\]"
        with self.assertRaisesRegex(Exception, exception):
            Dynamic(self.dmap, streams=[PointerX])

    def test_dynamic_util_inherits_dim_streams_clash_dict(self):
        exception = "The supplied stream objects PointerX\\(x=None\\) and PointerX\\(x=0\\) clash on the following parameters: \\['x'\\]"
        with self.assertRaisesRegex(Exception, exception):
            Dynamic(self.dmap, streams=dict(x=PointerX.param.x))