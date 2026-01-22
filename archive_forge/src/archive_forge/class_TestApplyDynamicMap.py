import numpy as np
import pandas as pd
import param
import pytest
from panel.widgets import IntSlider, RadioButtonGroup, TextInput
from holoviews import Dataset, util
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import Curve, Image
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import ParamMethod, Params
class TestApplyDynamicMap(ComparisonTestCase):

    def setUp(self):
        self.element = Curve([1, 2, 3])
        self.dmap_unsampled = DynamicMap(lambda i: Curve([0, 1, i]), kdims='Y')
        self.dmap = self.dmap_unsampled.redim.values(Y=[0, 1, 2])

    def test_dmap_apply_not_dynamic_unsampled(self):
        with self.assertRaises(ValueError):
            self.dmap_unsampled.apply(lambda x: x.relabel('Test'), dynamic=False)

    def test_dmap_apply_not_dynamic(self):
        applied = self.dmap.apply(lambda x: x.relabel('Test'), dynamic=False)
        self.assertEqual(applied, HoloMap(self.dmap[[0, 1, 2]]).relabel('Test'))

    def test_dmap_apply_not_dynamic_with_kwarg(self):
        applied = self.dmap.apply(lambda x, label: x.relabel(label), dynamic=False, label='Test')
        self.assertEqual(applied, HoloMap(self.dmap[[0, 1, 2]]).relabel('Test'))

    def test_dmap_apply_not_dynamic_with_instance_param(self):
        pinst = ParamClass()
        applied = self.dmap.apply(lambda x, label: x.relabel(label), label=pinst.param.label, dynamic=False)
        self.assertEqual(applied, HoloMap(self.dmap[[0, 1, 2]]).relabel('Test'))

    def test_dmap_apply_not_dynamic_with_param_method(self):
        pinst = ParamClass()
        applied = self.dmap.apply(lambda x, label: x.relabel(label), label=pinst.dynamic_label, dynamic=False)
        self.assertEqual(applied, HoloMap(self.dmap[[0, 1, 2]]).relabel('Test!'))

    def test_dmap_apply_dynamic(self):
        applied = self.dmap.apply(lambda x: x.relabel('Test'))
        self.assertEqual(len(applied.streams), 0)
        self.assertEqual(applied[1], self.dmap[1].relabel('Test'))

    def test_element_apply_method_as_string(self):
        applied = self.dmap.apply('relabel', label='Test')
        self.assertEqual(applied[1], self.dmap[1].relabel('Test'))

    def test_dmap_apply_dynamic_with_kwarg(self):
        applied = self.dmap.apply(lambda x, label: x.relabel(label), label='Test')
        self.assertEqual(len(applied.streams), 0)
        self.assertEqual(applied[1], self.dmap[1].relabel('Test'))

    def test_dmap_apply_dynamic_with_instance_param(self):
        pinst = ParamClass()
        applied = self.dmap.apply(lambda x, label: x.relabel(label), label=pinst.param.label)
        self.assertEqual(len(applied.streams), 1)
        stream = applied.streams[0]
        self.assertIsInstance(stream, Params)
        self.assertEqual(stream.parameterized, pinst)
        self.assertEqual(stream.parameters, [pinst.param.label])
        self.assertEqual(applied[1], self.dmap[1].relabel('Test'))
        pinst.label = 'Another label'
        self.assertEqual(applied[1], self.dmap[1].relabel('Another label'))

    def test_dmap_apply_method_as_string_with_instance_param(self):
        pinst = ParamClass()
        applied = self.dmap.apply('relabel', label=pinst.param.label)
        self.assertEqual(applied[1], self.dmap[1].relabel('Test'))
        pinst.label = 'Another label'
        self.assertEqual(applied[1], self.dmap[1].relabel('Another label'))

    def test_dmap_apply_param_method_with_dependencies(self):
        pinst = ParamClass()
        applied = self.dmap.apply(pinst.apply_label)
        self.assertEqual(len(applied.streams), 1)
        stream = applied.streams[0]
        self.assertIsInstance(stream, ParamMethod)
        self.assertEqual(stream.parameterized, pinst)
        self.assertEqual(stream.parameters, [pinst.param.label])
        self.assertEqual(applied[1], self.dmap[1].relabel('Test'))
        pinst.label = 'Another label'
        self.assertEqual(applied[1], self.dmap[1].relabel('Another label'))

    def test_dmap_apply_dynamic_with_param_method(self):
        pinst = ParamClass()
        applied = self.dmap.apply(lambda x, label: x.relabel(label), label=pinst.dynamic_label)
        self.assertEqual(len(applied.streams), 1)
        stream = applied.streams[0]
        self.assertIsInstance(stream, ParamMethod)
        self.assertEqual(stream.parameterized, pinst)
        self.assertEqual(stream.parameters, [pinst.param.label])
        self.assertEqual(applied[1], self.dmap[1].relabel('Test!'))
        pinst.label = 'Another label'
        self.assertEqual(applied[1], self.dmap[1].relabel('Another label!'))