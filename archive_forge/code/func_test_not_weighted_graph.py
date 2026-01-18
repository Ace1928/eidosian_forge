import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_not_weighted_graph(self):
    self.tsp(self.unweightedUG, 'greedy')
    self.tsp(self.unweightedDG, 'greedy')