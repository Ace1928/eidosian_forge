import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_plot_graph_annotation_overlay(self):
    plot = bokeh_renderer.get_plot(VLine(0) * self.graph)
    x_range = plot.handles['x_range']
    y_range = plot.handles['x_range']
    self.assertEqual(x_range.start, -1)
    self.assertEqual(x_range.end, 1)
    self.assertEqual(y_range.start, -1)
    self.assertEqual(y_range.end, 1)