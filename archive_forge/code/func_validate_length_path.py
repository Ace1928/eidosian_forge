import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def validate_length_path(G, s, t, soln_len, length, path):
    assert soln_len == length
    validate_path(G, s, t, length, path)