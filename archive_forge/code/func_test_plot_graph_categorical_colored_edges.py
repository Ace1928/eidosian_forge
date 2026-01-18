import numpy as np
import pytest
from matplotlib.collections import LineCollection, PolyCollection
from packaging.version import Version
from holoviews.core.data import Dataset
from holoviews.core.options import AbbreviatedException, Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Chord, Graph, Nodes, TriMesh, circular_layout
from holoviews.util.transform import dim
from .test_plot import TestMPLPlot, mpl_renderer
def test_plot_graph_categorical_colored_edges(self):
    g = self.graph3.opts(edge_color_index='start', edge_cmap=['#FFFFFF', '#000000'])
    plot = mpl_renderer.get_plot(g)
    edges = plot.handles['edges']
    colors = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    self.assertEqual(edges.get_colors(), colors)