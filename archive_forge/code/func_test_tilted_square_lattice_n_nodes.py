import itertools
from unittest.mock import MagicMock
import cirq
import networkx as nx
import pytest
from cirq import (
def test_tilted_square_lattice_n_nodes():
    for width, height in itertools.product(list(range(1, 4 + 1)), repeat=2):
        topo = TiltedSquareLattice(width, height)
        assert topo.n_nodes == topo.graph.number_of_nodes()