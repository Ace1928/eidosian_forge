from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
def test_trimesh_constructor_tuple_nodes(self):
    nodes = tuple(zip(*self.nodes))[:2]
    trimesh = TriMesh((self.simplices, nodes))
    self.assertEqual(trimesh.array(), np.array(self.simplices))
    self.assertEqual(trimesh.nodes.array([0, 1]), np.array(nodes).T)
    self.assertEqual(trimesh.nodes.dimension_values(2), np.arange(4))