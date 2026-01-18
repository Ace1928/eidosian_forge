from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
def test_trimesh_select(self):
    trimesh = TriMesh((self.simplices, self.nodes)).select(x=(0.1, None))
    self.assertEqual(trimesh.array(), np.array(self.simplices[1:]))