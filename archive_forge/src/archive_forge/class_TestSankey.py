from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
class TestSankey(ComparisonTestCase):

    def test_single_edge_sankey(self):
        sankey = Sankey([('A', 'B', 1)])
        links = list(sankey._sankey['links'])
        self.assertEqual(len(links), 1)
        del links[0]['source']['sourceLinks']
        del links[0]['target']['targetLinks']
        link = {'index': 0, 'source': {'index': 'A', 'values': (), 'targetLinks': [], 'value': 1, 'depth': 0, 'height': 1, 'column': 0, 'x0': 0, 'x1': 15, 'y0': 0.0, 'y1': 500.0}, 'target': {'index': 'B', 'values': (), 'sourceLinks': [], 'value': 1, 'depth': 1, 'height': 0, 'column': 1, 'x0': 985.0, 'x1': 1000.0, 'y0': 0.0, 'y1': 500.0}, 'value': 1, 'width': 500.0, 'y0': 250.0, 'y1': 250.0}
        self.assertEqual(links[0], link)