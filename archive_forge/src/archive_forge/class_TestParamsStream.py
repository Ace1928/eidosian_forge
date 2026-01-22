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
class TestParamsStream(LoggingComparisonTestCase):

    def setUp(self):
        super().setUp()

        class Inner(param.Parameterized):
            x = param.Number(default=0)
            y = param.Number(default=0)

        class InnerAction(Inner):
            action = param.Action(default=lambda o: o.param.trigger('action'))
        self.inner = Inner
        self.inner_action = InnerAction

    def test_param_stream_class(self):
        stream = Params(self.inner)
        self.assertEqual(set(stream.parameters), {self.inner.param.x, self.inner.param.y})
        self.assertEqual(stream.contents, {'x': 0, 'y': 0})
        values = []

        def subscriber(**kwargs):
            values.append(kwargs)
        stream.add_subscriber(subscriber)
        self.inner.x = 1
        self.assertEqual(values, [{'x': 1, 'y': 0}])

    def test_param_stream_instance(self):
        inner = self.inner(x=2)
        stream = Params(inner)
        self.assertEqual(set(stream.parameters), {inner.param.x, inner.param.y})
        self.assertEqual(stream.contents, {'x': 2, 'y': 0})
        values = []

        def subscriber(**kwargs):
            values.append(kwargs)
        stream.add_subscriber(subscriber)
        inner.y = 2
        self.assertEqual(values, [{'x': 2, 'y': 2}])

    def test_param_stream_instance_separate_parameters(self):
        inner = self.inner()
        xparam = Params(inner, ['x'])
        yparam = Params(inner, ['y'])
        valid, invalid = Stream._process_streams([xparam, yparam])
        self.assertEqual(len(valid), 2)
        self.assertEqual(len(invalid), 0)

    def test_param_stream_instance_overlapping_parameters(self):
        inner = self.inner()
        params1 = Params(inner)
        params2 = Params(inner)
        Stream._process_streams([params1, params2])
        self.log_handler.assertContains('WARNING', "['x', 'y']")

    def test_param_parameter_instance_separate_parameters(self):
        inner = self.inner()
        valid, invalid = Stream._process_streams([inner.param.x, inner.param.y])
        xparam, yparam = valid
        self.assertIs(xparam.parameterized, inner)
        self.assertEqual(xparam.parameters, [inner.param.x])
        self.assertIs(yparam.parameterized, inner)
        self.assertEqual(yparam.parameters, [inner.param.y])

    def test_param_parameter_instance_overlapping_parameters(self):
        inner = self.inner()
        Stream._process_streams([inner.param.x, inner.param.x])
        self.log_handler.assertContains('WARNING', "['x']")

    def test_param_stream_parameter_override(self):
        inner = self.inner(x=2)
        stream = Params(inner, parameters=['x'])
        self.assertEqual(stream.parameters, [inner.param.x])
        self.assertEqual(stream.contents, {'x': 2})
        values = []

        def subscriber(**kwargs):
            values.append(kwargs)
        stream.add_subscriber(subscriber)
        inner.x = 3
        self.assertEqual(values, [{'x': 3}])

    def test_param_stream_rename(self):
        inner = self.inner(x=2)
        stream = Params(inner, rename={'x': 'X', 'y': 'Y'})
        self.assertEqual(set(stream.parameters), {inner.param.x, inner.param.y})
        self.assertEqual(stream.contents, {'X': 2, 'Y': 0})
        values = []

        def subscriber(**kwargs):
            values.append(kwargs)
        stream.add_subscriber(subscriber)
        inner.y = 2
        self.assertEqual(values, [{'X': 2, 'Y': 2}])

    def test_param_stream_action(self):
        inner = self.inner_action()
        stream = Params(inner, ['action'])
        self.assertEqual(set(stream.parameters), {inner.param.action})
        values = []

        def subscriber(**kwargs):
            values.append(kwargs)
            self.assertEqual(set(stream.hashkey), {f'{id(inner)} action', '_memoize_key'})
        stream.add_subscriber(subscriber)
        inner.action(inner)
        self.assertEqual(values, [{'action': inner.action}])

    def test_param_stream_memoization(self):
        inner = self.inner_action()
        stream = Params(inner, ['action', 'x'])
        self.assertEqual(set(stream.parameters), {inner.param.action, inner.param.x})
        values = []

        def subscriber(**kwargs):
            values.append(kwargs)
            self.assertEqual(set(stream.hashkey), {f'{id(inner)} action', f'{id(inner)} x', '_memoize_key'})
        stream.add_subscriber(subscriber)
        inner.action(inner)
        inner.x = 0
        self.assertEqual(values, [{'action': inner.action, 'x': 0}])

    def test_params_stream_batch_watch(self):
        tap = Tap(x=0, y=1)
        params = Params(parameters=[tap.param.x, tap.param.y])
        values = []

        def subscriber(**kwargs):
            values.append(kwargs)
        params.add_subscriber(subscriber)
        tap.param.trigger('x', 'y')
        assert values == [{'x': 0, 'y': 1}]
        tap.event(x=1, y=2)
        assert values == [{'x': 0, 'y': 1}, {'x': 1, 'y': 2}]

    def test_params_no_names(self):
        a = IntSlider()
        b = IntSlider()
        p = Params(parameters=[a.param.value, b.param.value])
        assert len(p.hashkey) == 3

    def test_params_identical_names(self):
        a = IntSlider(name='Name')
        b = IntSlider(name='Name')
        p = Params(parameters=[a.param.value, b.param.value])
        assert len(p.hashkey) == 3