import numpy as np
import panel as pn
from bokeh.models import FactorRange, FixedTicker, HoverTool, Range1d, Span
from holoviews.core import DynamicMap, HoloMap, NdOverlay, Overlay
from holoviews.element import (
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import Stream, Tap
from holoviews.util import Dynamic
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
class TestLegends(TestBokehPlot):

    def test_overlay_legend(self):
        overlay = Curve(range(10), label='A') * Curve(range(10), label='B')
        plot = bokeh_renderer.get_plot(overlay)
        legend_labels = [l.label['value'] for l in plot.state.legend[0].items]
        self.assertEqual(legend_labels, ['A', 'B'])

    def test_overlay_legend_with_labels(self):
        overlay = (Curve(range(10), label='A') * Curve(range(10), label='B')).opts(legend_labels={'A': 'A Curve', 'B': 'B Curve'})
        plot = bokeh_renderer.get_plot(overlay)
        legend_labels = [l.label['value'] for l in plot.state.legend[0].items]
        self.assertEqual(legend_labels, ['A Curve', 'B Curve'])

    def test_holomap_legend_updates(self):
        hmap = HoloMap({i: Curve([1, 2, 3], label=chr(65 + i + 2)) * Curve([1, 2, 3], label='B') for i in range(3)})
        plot = bokeh_renderer.get_plot(hmap)
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'C'}, {'value': 'B'}])
        plot.update((1,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'B'}, {'value': 'D'}])
        plot.update((2,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'B'}, {'value': 'E'}])

    def test_holomap_legend_updates_varying_lengths(self):
        hmap = HoloMap({i: Overlay([Curve([1, 2, j], label=chr(65 + j)) for j in range(i)]) for i in range(1, 4)})
        plot = bokeh_renderer.get_plot(hmap)
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'A'}])
        plot.update((2,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'A'}, {'value': 'B'}])
        plot.update((3,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'A'}, {'value': 'B'}, {'value': 'C'}])

    def test_dynamicmap_legend_updates(self):
        hmap = HoloMap({i: Curve([1, 2, 3], label=chr(65 + i + 2)) * Curve([1, 2, 3], label='B') for i in range(3)})
        dmap = Dynamic(hmap)
        plot = bokeh_renderer.get_plot(dmap)
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'C'}, {'value': 'B'}])
        plot.update((1,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'B'}, {'value': 'D'}])
        plot.update((2,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'B'}, {'value': 'E'}])

    def test_dynamicmap_legend_updates_add_dynamic_plots(self):
        hmap = HoloMap({i: Overlay([Curve([1, 2, j], label=chr(65 + j)) for j in range(i)]) for i in range(1, 4)})
        dmap = Dynamic(hmap)
        plot = bokeh_renderer.get_plot(dmap)
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'A'}])
        plot.update((2,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'A'}, {'value': 'B'}])
        plot.update((3,))
        legend_labels = [property_to_dict(item.label) for item in plot.state.legend[0].items]
        self.assertEqual(legend_labels, [{'value': 'A'}, {'value': 'B'}, {'value': 'C'}])

    def test_dynamicmap_ndoverlay_shrink_number_of_items(self):
        selected = Stream.define('selected', items=3)()

        def callback(items):
            return NdOverlay({j: Overlay([Curve([1, 2, j])]) for j in range(items)})
        dmap = DynamicMap(callback, streams=[selected])
        plot = bokeh_renderer.get_plot(dmap)
        selected.event(items=2)
        self.assertEqual(len([r for r in plot.state.renderers if r.visible]), 2)

    def test_dynamicmap_variable_length_overlay(self):
        selected = Stream.define('selected', items=[1])()

        def callback(items):
            return Overlay([Box(0, 0, radius * 2) for radius in items])
        dmap = DynamicMap(callback, streams=[selected])
        plot = bokeh_renderer.get_plot(dmap)
        assert len(plot.subplots) == 1
        selected.event(items=[1, 2, 4])
        assert len(plot.subplots) == 3
        selected.event(items=[1, 4])
        sp1, sp2, sp3 = plot.subplots.values()
        assert sp1.handles['cds'].data['xs'][0].min() == -1
        assert sp1.handles['glyph_renderer'].visible
        assert sp2.handles['cds'].data['xs'][0].min() == -4
        assert sp2.handles['glyph_renderer'].visible
        assert sp3.handles['cds'].data['xs'][0].min() == -4
        assert not sp3.handles['glyph_renderer'].visible