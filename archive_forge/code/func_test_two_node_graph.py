import math
import pytest
import networkx as nx
from networkx.algorithms.planar_drawing import triangulate_embedding
def test_two_node_graph():
    embedding_data = {0: [1], 1: [0]}
    check_embedding_data(embedding_data)