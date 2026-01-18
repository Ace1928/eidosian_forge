import testtools
import testresources
from testresources import split_by_resources, _resource_graph
from testresources.tests import ResultWithResourceExtensions
import unittest
def test_wikipedia_example(self):
    """Performing KruskalsMST on a graph returns a spanning tree.

        See http://en.wikipedia.org/wiki/Kruskal%27s_algorithm.
        """
    A = 'A'
    B = 'B'
    C = 'C'
    D = 'D'
    E = 'E'
    F = 'F'
    G = 'G'
    graph = {A: {B: 7, D: 5}, B: {A: 7, C: 8, D: 9, E: 7}, C: {B: 8, E: 5}, D: {A: 5, B: 9, E: 15, F: 6}, E: {B: 7, C: 5, D: 15, F: 8, G: 9}, F: {D: 6, E: 8, G: 11}, G: {E: 9, F: 11}}
    expected = {A: {B: 7, D: 5}, B: {A: 7, E: 7}, C: {E: 5}, D: {A: 5, F: 6}, E: {B: 7, C: 5, G: 9}, F: {D: 6}, G: {E: 9}}
    result = testresources._kruskals_graph_MST(graph)
    e_weight = sum((sum(row.values()) for row in expected.values()))
    r_weight = sum((sum(row.values()) for row in result.values()))
    self.assertEqual(e_weight, r_weight)
    self.assertEqual(expected, testresources._kruskals_graph_MST(graph))