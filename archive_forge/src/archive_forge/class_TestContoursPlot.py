import datetime as dt
import numpy as np
import pandas as pd
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core import HoloMap, NdOverlay
from holoviews.core.options import Cycle
from holoviews.element import Contours, Path, Polygons
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import PolyDraw
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
class TestContoursPlot(TestBokehPlot):

    def test_empty_contours_plot(self):
        contours = Contours([], vdims=['Intensity'])
        plot = bokeh_renderer.get_plot(contours)
        source = plot.handles['source']
        self.assertEqual(len(source.data['xs']), 0)
        self.assertEqual(len(source.data['ys']), 0)
        self.assertEqual(len(source.data['Intensity']), 0)

    def test_contours_color_op(self):
        contours = Contours([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 'green'}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 'red'}], vdims='color').opts(color='color')
        plot = bokeh_renderer.get_plot(contours)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color'})
        self.assertEqual(cds.data['color'], np.array(['green', 'red']))

    def test_contours_linear_color_op(self):
        contours = Contours([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 7}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 3}], vdims='color').opts(color='color')
        plot = bokeh_renderer.get_plot(contours)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color', 'transform': cmapper})
        self.assertEqual(cds.data['color'], np.array([7, 3]))
        self.assertIsInstance(cmapper, LinearColorMapper)
        self.assertEqual(cmapper.low, 3)
        self.assertEqual(cmapper.high, 7)

    def test_contours_empty_path(self):
        contours = Contours([pd.DataFrame([], columns=['x', 'y', 'color', 'line_width']), pd.DataFrame({'x': np.random.rand(10), 'y': np.random.rand(10), 'color': ['red'] * 10, 'line_width': [3] * 10}, columns=['x', 'y', 'color', 'line_width'])], vdims=['color', 'line_width']).opts(color='color', line_width='line_width')
        plot = bokeh_renderer.get_plot(contours)
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.line_color, 'red')
        self.assertEqual(glyph.line_width, 3)

    def test_contours_linear_color_op_update(self):
        contours = HoloMap({0: Contours([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 7}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 3}], vdims='color'), 1: Contours([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 5}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 2}], vdims='color')}).opts(color='color', framewise=True)
        plot = bokeh_renderer.get_plot(contours)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        plot.update((0,))
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color', 'transform': cmapper})
        self.assertEqual(cds.data['color'], np.array([7, 3]))
        self.assertEqual(cmapper.low, 3)
        self.assertEqual(cmapper.high, 7)
        plot.update((1,))
        self.assertEqual(cds.data['color'], np.array([5, 2]))
        self.assertEqual(cmapper.low, 2)
        self.assertEqual(cmapper.high, 5)

    def test_contours_categorical_color_op(self):
        contours = Contours([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 'b'}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 'a'}], vdims='color').opts(color='color')
        plot = bokeh_renderer.get_plot(contours)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color', 'transform': cmapper})
        self.assertEqual(cds.data['color'], np.array(['b', 'a']))
        self.assertIsInstance(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['b', 'a'])

    def test_contours_alpha_op(self):
        contours = Contours([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'alpha': 0.7}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'alpha': 0.3}], vdims='alpha').opts(alpha='alpha')
        plot = bokeh_renderer.get_plot(contours)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'alpha'})
        self.assertEqual(cds.data['alpha'], np.array([0.7, 0.3]))

    def test_contours_line_width_op(self):
        contours = Contours([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'line_width': 7}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'line_width': 3}], vdims='line_width').opts(line_width='line_width')
        plot = bokeh_renderer.get_plot(contours)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        self.assertEqual(property_to_dict(glyph.line_width), {'field': 'line_width'})
        self.assertEqual(cds.data['line_width'], np.array([7, 3]))