from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
class ChordTests(ComparisonTestCase):

    def setUp(self):
        self.simplices = [(0, 1, 2), (1, 2, 3), (2, 3, 4)]

    def test_chord_constructor_no_vdims(self):
        chord = Chord(self.simplices)
        nodes = np.array([[0.8660254037844387, 0.49999999999999994, 0], [-0.4999999999999998, 0.8660254037844388, 1], [-0.5000000000000004, -0.8660254037844384, 2], [0.8660254037844379, -0.5000000000000012, 3]])
        self.assertEqual(chord.nodes, Nodes(nodes))
        self.assertEqual(chord.array(), np.array([s[:2] for s in self.simplices]))

    def test_chord_constructor_with_vdims(self):
        chord = Chord(self.simplices, vdims=['z'])
        nodes = np.array([[0.9396926207859084, 0.3420201433256687, 0], [6.123233995736766e-17, 1.0, 1], [-0.8660254037844388, -0.4999999999999998, 2], [0.7660444431189779, -0.6427876096865396, 3]])
        self.assertEqual(chord.nodes, Nodes(nodes))
        self.assertEqual(chord.array(), np.array(self.simplices))

    def test_chord_constructor_self_reference(self):
        chord = Chord([('A', 'B', 2), ('B', 'A', 3), ('A', 'A', 2)])
        nodes = pd.DataFrame([[-0.5, 0.866025, 'A'], [0.5, -0.866025, 'B']], columns=['x', 'y', 'index'])
        self.assertEqual(chord.nodes, Nodes(nodes))