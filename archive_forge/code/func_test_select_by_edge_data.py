from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
def test_select_by_edge_data(self):
    graph = Graph(((self.target, self.source, self.edge_info),), vdims=['info'])
    selection = Graph(([(0, 0, 0), (0, 1, 1)], list(zip(*self.nodes))[:2]), vdims=['info'])
    self.assertEqual(graph.select(info=(0, 2)), selection)