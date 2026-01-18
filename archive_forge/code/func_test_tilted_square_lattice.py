import itertools
from unittest.mock import MagicMock
import cirq
import networkx as nx
import pytest
from cirq import (
@pytest.mark.parametrize('width, height', list(itertools.product([1, 2, 3, 24], repeat=2)))
def test_tilted_square_lattice(width, height):
    topo = TiltedSquareLattice(width, height)
    assert topo.graph.number_of_edges() == width * height
    assert all((1 <= topo.graph.degree[node] <= 4 for node in topo.graph.nodes))
    assert topo.name == f'tilted-square-lattice-{width}-{height}'
    assert topo.n_nodes == topo.graph.number_of_nodes()
    assert nx.is_connected(topo.graph)
    assert nx.algorithms.planarity.check_planarity(topo.graph)
    cirq.testing.assert_equivalent_repr(topo)