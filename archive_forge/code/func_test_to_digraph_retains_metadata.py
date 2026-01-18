import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_to_digraph_retains_metadata(self):
    root = tree.Node('chickens', alive=True)
    dead_chicken = tree.Node('chicken.1', alive=False)
    root.add(dead_chicken)
    g = root.to_digraph()
    self.assertEqual(g.nodes['chickens'], {'alive': True})
    self.assertEqual(g.nodes['chicken.1'], {'alive': False})