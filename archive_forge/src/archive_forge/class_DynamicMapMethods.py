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
class DynamicMapMethods(ComparisonTestCase):

    def test_deep_relabel_label(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=['i']).relabel(label='Test')
        self.assertEqual(dmap[0].label, 'Test')

    def test_deep_relabel_group(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=['i']).relabel(group='Test')
        self.assertEqual(dmap[0].group, 'Test')

    def test_redim_dimension_name(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=['i']).redim(i='New')
        self.assertEqual(dmap.kdims[0].name, 'New')

    def test_redim_dimension_range_aux(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=['i']).redim.range(i=(0, 1))
        self.assertEqual(dmap.kdims[0].range, (0, 1))

    def test_redim_dimension_values_cache_reset_1D(self):
        fn = lambda i: Curve([i, i])
        dmap = DynamicMap(fn, kdims=['i'])[{0, 1, 2, 3, 4, 5}]
        self.assertEqual(dmap.keys(), [0, 1, 2, 3, 4, 5])
        redimmed = dmap.redim.values(i=[2, 3, 5, 6, 8])
        self.assertEqual(redimmed.keys(), [2, 3, 5])

    def test_redim_dimension_values_cache_reset_2D_single(self):
        fn = lambda i, j: Curve([i, j])
        keys = [(0, 1), (1, 0), (2, 2), (2, 5), (3, 3)]
        dmap = DynamicMap(fn, kdims=['i', 'j'])[keys]
        self.assertEqual(dmap.keys(), keys)
        redimmed = dmap.redim.values(i=[2, 10, 50])
        self.assertEqual(redimmed.keys(), [(2, 2), (2, 5)])

    def test_redim_dimension_values_cache_reset_2D_multi(self):
        fn = lambda i, j: Curve([i, j])
        keys = [(0, 1), (1, 0), (2, 2), (2, 5), (3, 3)]
        dmap = DynamicMap(fn, kdims=['i', 'j'])[keys]
        self.assertEqual(dmap.keys(), keys)
        redimmed = dmap.redim.values(i=[2, 10, 50], j=[5, 50, 100])
        self.assertEqual(redimmed.keys(), [(2, 5)])

    def test_redim_dimension_unit_aux(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=['i']).redim.unit(i='m/s')
        self.assertEqual(dmap.kdims[0].unit, 'm/s')

    def test_redim_dimension_type_aux(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=['i']).redim.type(i=int)
        self.assertEqual(dmap.kdims[0].type, int)

    def test_deep_redim_dimension_name(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=['i']).redim(x='X')
        self.assertEqual(dmap[0].kdims[0].name, 'X')

    def test_deep_redim_dimension_name_with_spec(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=['i']).redim(Image, x='X')
        self.assertEqual(dmap[0].kdims[0].name, 'X')

    def test_deep_getitem_bounded_kdims(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap[:, 5:10][10], fn(10)[5:10])

    def test_deep_getitem_bounded_kdims_and_vdims(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap[:, 5:10, 0:5][10], fn(10)[5:10, 0:5])

    def test_deep_getitem_cross_product_and_slice(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap[[10, 11, 12], 5:10], dmap.clone([(i, fn(i)[5:10]) for i in range(10, 13)]))

    def test_deep_getitem_index_and_slice(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap[10, 5:10], fn(10)[5:10])

    def test_deep_getitem_cache_sliced(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        dmap[10]
        self.assertEqual(dmap[:, 5:10][10], fn(10)[5:10])

    def test_deep_select_slice_kdim(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap.select(x=(5, 10))[10], fn(10)[5:10])

    def test_deep_select_slice_kdim_and_vdims(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap.select(x=(5, 10), y=(0, 5))[10], fn(10)[5:10, 0:5])

    def test_deep_select_slice_kdim_no_match(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap.select(DynamicMap, x=(5, 10))[10], fn(10))

    def test_deep_apply_element_function(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        mapped = dmap.apply(lambda x: x.clone(x.data * 2))
        curve = fn(10)
        self.assertEqual(mapped[10], curve.clone(curve.data * 2))

    def test_deep_apply_element_param_function(self):
        fn = lambda i: Curve(np.arange(i))

        class Test(param.Parameterized):
            a = param.Integer(default=1)
        test = Test()

        @param.depends(test.param.a)
        def op(obj, a):
            return obj.clone(obj.data * 2)
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        mapped = dmap.apply(op)
        test.a = 2
        curve = fn(10)
        self.assertEqual(mapped[10], curve.clone(curve.data * 2))

    def test_deep_apply_element_function_with_kwarg(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        mapped = dmap.apply(lambda x, label: x.relabel(label), label='New label')
        self.assertEqual(mapped[10], fn(10).relabel('New label'))

    def test_deep_map_apply_element_function_with_stream_kwarg(self):
        stream = Stream.define('Test', label='New label')()
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        mapped = dmap.apply(lambda x, label: x.relabel(label), streams=[stream])
        self.assertEqual(mapped[10], fn(10).relabel('New label'))

    def test_deep_map_apply_parameterized_method_with_stream_kwarg(self):

        class Test(param.Parameterized):
            label = param.String(default='label')

            @param.depends('label')
            def value(self):
                return self.label.title()
        test = Test()
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        mapped = dmap.apply(lambda x, label: x.relabel(label), label=test.value)
        curve = fn(10)
        self.assertEqual(mapped[10], curve.relabel('Label'))
        test.label = 'new label'
        self.assertEqual(mapped[10], curve.relabel('New Label'))

    def test_deep_apply_parameterized_method_with_dependency(self):

        class Test(param.Parameterized):
            label = param.String(default='label')

            @param.depends('label')
            def relabel(self, obj):
                return obj.relabel(self.label.title())
        test = Test()
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        mapped = dmap.apply(test.relabel)
        curve = fn(10)
        self.assertEqual(mapped[10], curve.relabel('Label'))
        test.label = 'new label'
        self.assertEqual(mapped[10], curve.relabel('New Label'))

    def test_deep_apply_parameterized_method_with_dependency_and_static_kwarg(self):

        class Test(param.Parameterized):
            label = param.String(default='label')

            @param.depends('label')
            def relabel(self, obj, group):
                return obj.relabel(self.label.title(), group)
        test = Test()
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        mapped = dmap.apply(test.relabel, group='Group')
        curve = fn(10)
        self.assertEqual(mapped[10], curve.relabel('Label', 'Group'))
        test.label = 'new label'
        self.assertEqual(mapped[10], curve.relabel('New Label', 'Group'))

    def test_deep_map_transform_element_type(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        dmap[10]
        mapped = dmap.map(lambda x: Scatter(x), Curve)
        area = mapped[11]
        self.assertEqual(area, Scatter(fn(11)))

    def test_deep_apply_transform_element_type(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        dmap[10]
        mapped = dmap.apply(lambda x: Scatter(x))
        area = mapped[11]
        self.assertEqual(area, Scatter(fn(11)))

    def test_deep_map_apply_dmap_function(self):
        fn = lambda i: Curve(np.arange(i))
        dmap1 = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        dmap2 = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        mapped = (dmap1 + dmap2).map(lambda x: x[10], DynamicMap)
        self.assertEqual(mapped, Layout([('DynamicMap.I', fn(10)), ('DynamicMap.II', fn(10))]))

    def test_deep_map_apply_dmap_function_no_clone(self):
        fn = lambda i: Curve(np.arange(i))
        dmap1 = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        dmap2 = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        layout = dmap1 + dmap2
        mapped = layout.map(lambda x: x[10], DynamicMap, clone=False)
        self.assertIs(mapped, layout)

    def test_dynamic_reindex_reorder(self):
        history = deque(maxlen=10)

        def history_callback(x, y):
            history.append((x, y))
            return Points(list(history))
        dmap = DynamicMap(history_callback, kdims=['x', 'y'])
        reindexed = dmap.reindex(['y', 'x'])
        points = reindexed[2, 1]
        self.assertEqual(points, Points([(1, 2)]))

    def test_dynamic_reindex_drop_raises_exception(self):
        history = deque(maxlen=10)

        def history_callback(x, y):
            history.append((x, y))
            return Points(list(history))
        dmap = DynamicMap(history_callback, kdims=['x', 'y'])
        exception = 'DynamicMap does not allow dropping dimensions, reindex may only be used to reorder dimensions.'
        with self.assertRaisesRegex(ValueError, exception):
            dmap.reindex(['x'])

    def test_dynamic_groupby_kdims_and_streams(self):

        def plot_function(mydim, data):
            return Scatter(data[data[:, 2] == mydim])
        buff = Buffer(data=np.empty((0, 3)))
        dmap = DynamicMap(plot_function, streams=[buff], kdims='mydim').redim.values(mydim=[0, 1, 2])
        ndlayout = dmap.groupby('mydim', container_type=NdLayout)
        self.assertIsInstance(ndlayout[0], DynamicMap)
        data = np.array([(0, 0, 0), (1, 1, 1), (2, 2, 2)])
        buff.send(data)
        self.assertIs(ndlayout[0].callback.inputs[0], dmap)
        self.assertIs(ndlayout[1].callback.inputs[0], dmap)
        self.assertIs(ndlayout[2].callback.inputs[0], dmap)
        self.assertEqual(ndlayout[0][()], Scatter([(0, 0)]))
        self.assertEqual(ndlayout[1][()], Scatter([(1, 1)]))
        self.assertEqual(ndlayout[2][()], Scatter([(2, 2)]))

    def test_dynamic_split_overlays_on_ndoverlay(self):
        dmap = DynamicMap(lambda: NdOverlay({i: Points([i]) for i in range(3)}))
        initialize_dynamic(dmap)
        keys, dmaps = dmap._split_overlays()
        self.assertEqual(keys, [(0,), (1,), (2,)])
        self.assertEqual(dmaps[0][()], Points([0]))
        self.assertEqual(dmaps[1][()], Points([1]))
        self.assertEqual(dmaps[2][()], Points([2]))

    def test_dynamic_split_overlays_on_overlay(self):
        dmap1 = DynamicMap(lambda: Points([]))
        dmap2 = DynamicMap(lambda: Curve([]))
        dmap = dmap1 * dmap2
        initialize_dynamic(dmap)
        keys, dmaps = dmap._split_overlays()
        self.assertEqual(keys, [('Points', 'I'), ('Curve', 'I')])
        self.assertEqual(dmaps[0][()], Points([]))
        self.assertEqual(dmaps[1][()], Curve([]))

    def test_dynamic_split_overlays_on_varying_order_overlay(self):

        def cb(i):
            if i % 2 == 0:
                return Curve([]) * Points([])
            else:
                return Points([]) * Curve([])
        dmap = DynamicMap(cb, kdims='i').redim.range(i=(0, 4))
        initialize_dynamic(dmap)
        keys, dmaps = dmap._split_overlays()
        self.assertEqual(keys, [('Curve', 'I'), ('Points', 'I')])
        self.assertEqual(dmaps[0][0], Curve([]))
        self.assertEqual(dmaps[0][1], Curve([]))
        self.assertEqual(dmaps[1][0], Points([]))
        self.assertEqual(dmaps[1][1], Points([]))

    def test_dynamic_split_overlays_on_missing_item_in_overlay(self):

        def cb(i):
            if i % 2 == 0:
                return Curve([]) * Points([])
            else:
                return Scatter([]) * Curve([])
        dmap = DynamicMap(cb, kdims='i').redim.range(i=(0, 4))
        initialize_dynamic(dmap)
        keys, dmaps = dmap._split_overlays()
        self.assertEqual(keys, [('Curve', 'I'), ('Points', 'I')])
        self.assertEqual(dmaps[0][0], Curve([]))
        self.assertEqual(dmaps[0][1], Curve([]))
        self.assertEqual(dmaps[1][0], Points([]))
        with self.assertRaises(KeyError):
            dmaps[1][1]