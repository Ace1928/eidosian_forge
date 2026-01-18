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
def test_plot_graph_numerically_colored_nodes(self):
    g = self.graph3.opts(color_index='Weight', cmap='viridis')
    plot = mpl_renderer.get_plot(g)
    nodes = plot.handles['nodes']
    self.assertEqual(np.asarray(nodes.get_array()), self.weights)
    self.assertEqual(nodes.get_clim(), (self.weights.min(), self.weights.max()))