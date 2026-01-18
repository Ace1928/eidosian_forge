import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_to_diagraph(self):
    root = self._make_species()
    g = root.to_digraph()
    self.assertEqual(root.child_count(only_direct=False) + 1, len(g))
    for node in root.dfs_iter(include_self=True):
        self.assertIn(node.item, g)
    self.assertEqual([], list(g.predecessors('animal')))
    self.assertEqual(['animal'], list(g.predecessors('reptile')))
    self.assertEqual(['primate'], list(g.predecessors('human')))
    self.assertEqual(['mammal'], list(g.predecessors('primate')))
    self.assertEqual(['animal'], list(g.predecessors('mammal')))
    self.assertEqual(['mammal', 'reptile'], list(g.successors('animal')))