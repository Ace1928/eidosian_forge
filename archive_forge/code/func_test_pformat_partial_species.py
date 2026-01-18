import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_pformat_partial_species(self):
    root = self._make_species()
    expected = '\nreptile\n'
    self.assertEqual(expected.strip(), root[1].pformat())
    expected = '\nmammal\n|__horse\n|__primate\n   |__monkey\n   |__human\n'
    self.assertEqual(expected.strip(), root[0].pformat())
    expected = '\nprimate\n|__monkey\n|__human\n'
    self.assertEqual(expected.strip(), root[0][1].pformat())
    expected = '\nmonkey\n'
    self.assertEqual(expected.strip(), root[0][1][0].pformat())