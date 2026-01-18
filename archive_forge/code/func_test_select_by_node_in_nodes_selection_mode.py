from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
def test_select_by_node_in_nodes_selection_mode(self):
    graph = Graph(((self.source, self.source + 1), self.nodes))
    selection = Graph(([(1, 2)], list(zip(*self.nodes))[1:3]))
    self.assertEqual(graph.select(index=(1, 3), selection_mode='nodes'), selection)