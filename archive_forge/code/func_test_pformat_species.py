import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_pformat_species(self):
    root = self._make_species()
    expected = '\nanimal\n|__mammal\n|  |__horse\n|  |__primate\n|     |__monkey\n|     |__human\n|__reptile\n'
    self.assertEqual(expected.strip(), root.pformat())