import math
import pytest
import networkx as nx
from networkx.algorithms.planar_drawing import triangulate_embedding
def test_one_node_graph():
    embedding_data = {0: []}
    check_embedding_data(embedding_data)