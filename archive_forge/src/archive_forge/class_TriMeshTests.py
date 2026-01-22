from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
class TriMeshTests(ComparisonTestCase):

    def setUp(self):
        self.nodes = [(0, 0, 0), (0.5, 1, 1), (1.0, 0, 2), (1.5, 1, 4)]
        self.simplices = [(0, 1, 2), (1, 2, 3)]

    def test_trimesh_constructor(self):
        nodes = [n[:2] for n in self.nodes]
        trimesh = TriMesh((self.simplices, nodes))
        self.assertEqual(trimesh.array(), np.array(self.simplices))
        self.assertEqual(trimesh.nodes.array([0, 1]), np.array(nodes))
        self.assertEqual(trimesh.nodes.dimension_values(2), np.arange(4))

    def test_trimesh_empty(self):
        trimesh = TriMesh([])
        self.assertEqual(trimesh.array(), np.empty((0, 3)))
        self.assertEqual(trimesh.nodes.array(), np.empty((0, 3)))

    def test_trimesh_empty_clone(self):
        trimesh = TriMesh([]).clone()
        self.assertEqual(trimesh.array(), np.empty((0, 3)))
        self.assertEqual(trimesh.nodes.array(), np.empty((0, 3)))

    def test_trimesh_constructor_tuple_nodes(self):
        nodes = tuple(zip(*self.nodes))[:2]
        trimesh = TriMesh((self.simplices, nodes))
        self.assertEqual(trimesh.array(), np.array(self.simplices))
        self.assertEqual(trimesh.nodes.array([0, 1]), np.array(nodes).T)
        self.assertEqual(trimesh.nodes.dimension_values(2), np.arange(4))

    def test_trimesh_constructor_df_nodes(self):
        nodes_df = pd.DataFrame(self.nodes, columns=['x', 'y', 'z'])
        trimesh = TriMesh((self.simplices, nodes_df))
        nodes = Nodes([(0, 0, 0, 0), (0.5, 1, 1, 1), (1.0, 0, 2, 2), (1.5, 1, 3, 4)], vdims='z')
        self.assertEqual(trimesh.array(), np.array(self.simplices))
        self.assertEqual(trimesh.nodes, nodes)

    def test_trimesh_constructor_point_nodes(self):
        nodes = [n[:2] for n in self.nodes]
        trimesh = TriMesh((self.simplices, Points(self.nodes)))
        self.assertEqual(trimesh.array(), np.array(self.simplices))
        self.assertEqual(trimesh.nodes.array([0, 1]), np.array(nodes))
        self.assertEqual(trimesh.nodes.dimension_values(2), np.arange(4))

    def test_trimesh_edgepaths(self):
        trimesh = TriMesh((self.simplices, self.nodes))
        paths = [np.array([(0, 0), (0.5, 1), (1, 0), (0, 0), (np.nan, np.nan), (0.5, 1), (1, 0), (1.5, 1), (0.5, 1)])]
        for p1, p2 in zip(trimesh.edgepaths.split(datatype='array'), paths):
            self.assertEqual(p1, p2)

    def test_trimesh_select(self):
        trimesh = TriMesh((self.simplices, self.nodes)).select(x=(0.1, None))
        self.assertEqual(trimesh.array(), np.array(self.simplices[1:]))