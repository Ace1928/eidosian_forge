from operator import itemgetter
import networkx as nx
def validate_possible_communities(result, *expected):
    assert any((set_of_sets(result) == set_of_sets(p) for p in expected))