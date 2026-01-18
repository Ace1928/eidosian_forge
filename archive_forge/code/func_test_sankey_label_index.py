import numpy as np
import pandas as pd
from holoviews import render
from holoviews.core.data import Dataset, Dimension
from holoviews.element import Sankey
from .test_plot import TestBokehPlot, bokeh_renderer
def test_sankey_label_index(self):
    sankey = Sankey(([(0, 2, 5), (0, 3, 7), (0, 4, 6), (1, 2, 2), (1, 3, 9), (1, 4, 4)], Dataset(enumerate('ABXYZ'), 'index', 'label'))).opts(label_index='label', tools=['hover'])
    plot = bokeh_renderer.get_plot(sankey)
    scatter_source = plot.handles['scatter_1_source']
    text_source = plot.handles['text_1_source']
    patch_source = plot.handles['patches_1_source']
    scatter_index = np.arange(5)
    self.assertEqual(scatter_source.data['index'], scatter_index)
    text_data = {'x': np.array([18.75, 18.75, 1003.75, 1003.75, 1003.75]), 'y': np.array([125.454545, 375.454545, 48.787879, 229.090909, 430.30303]), 'text': ['A - 18', 'B - 15', 'X - 7', 'Y - 16', 'Z - 10']}
    for k in text_data:
        self.assertEqual(text_source.data[k], text_data[k])
    patch_data = {'start_values': ['A', 'A', 'A', 'B', 'B', 'B'], 'end_values': ['X', 'Y', 'Z', 'X', 'Y', 'Z'], 'Value': np.array([5, 7, 6, 2, 9, 4])}
    for k in patch_data:
        self.assertEqual(patch_source.data[k], patch_data[k])
    renderers = plot.state.renderers
    quad_renderer = plot.handles['quad_1_glyph_renderer']
    text_renderer = plot.handles['text_1_glyph_renderer']
    graph_renderer = plot.handles['glyph_renderer']
    self.assertTrue(renderers.index(graph_renderer) < renderers.index(quad_renderer))
    self.assertTrue(renderers.index(quad_renderer) < renderers.index(text_renderer))