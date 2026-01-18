import pytest
import networkx as nx
from networkx.algorithms.connectivity.kcomponents import (
def test_set_consolidation_rosettacode():

    def list_of_sets_equal(result, solution):
        assert {frozenset(s) for s in result} == {frozenset(s) for s in solution}
    question = [{'A', 'B'}, {'C', 'D'}]
    solution = [{'A', 'B'}, {'C', 'D'}]
    list_of_sets_equal(_consolidate(question, 1), solution)
    question = [{'A', 'B'}, {'B', 'C'}]
    solution = [{'A', 'B', 'C'}]
    list_of_sets_equal(_consolidate(question, 1), solution)
    question = [{'A', 'B'}, {'C', 'D'}, {'D', 'B'}]
    solution = [{'A', 'C', 'B', 'D'}]
    list_of_sets_equal(_consolidate(question, 1), solution)
    question = [{'H', 'I', 'K'}, {'A', 'B'}, {'C', 'D'}, {'D', 'B'}, {'F', 'G', 'H'}]
    solution = [{'A', 'C', 'B', 'D'}, {'G', 'F', 'I', 'H', 'K'}]
    list_of_sets_equal(_consolidate(question, 1), solution)
    question = [{'A', 'H'}, {'H', 'I', 'K'}, {'A', 'B'}, {'C', 'D'}, {'D', 'B'}, {'F', 'G', 'H'}]
    solution = [{'A', 'C', 'B', 'D', 'G', 'F', 'I', 'H', 'K'}]
    list_of_sets_equal(_consolidate(question, 1), solution)
    question = [{'H', 'I', 'K'}, {'A', 'B'}, {'C', 'D'}, {'D', 'B'}, {'F', 'G', 'H'}, {'A', 'H'}]
    solution = [{'A', 'C', 'B', 'D', 'G', 'F', 'I', 'H', 'K'}]
    list_of_sets_equal(_consolidate(question, 1), solution)