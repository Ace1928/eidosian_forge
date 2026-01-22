import itertools
from typing import Iterable
import cirq
import networkx as nx
def _gridqubits_to_graph_device(qubits: Iterable[cirq.GridQubit]):
    return nx.Graph((pair for pair in itertools.combinations(qubits, 2) if pair[0].is_adjacent(pair[1])))