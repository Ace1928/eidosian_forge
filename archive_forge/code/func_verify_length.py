import itertools
import pytest
import networkx as nx
def verify_length(coloring, expected):
    coloring = dict_to_sets(coloring)
    return len(coloring) == expected