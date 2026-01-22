from typing import Deque, Dict, List, Set, Tuple, TYPE_CHECKING
from collections import deque
import networkx as nx
from cirq.transformers.routing import initial_mapper
from cirq import protocols, value
def _closest_unmapped_qubit(self, source: 'cirq.Qid', mapped_physicals: Set['cirq.Qid']) -> 'cirq.Qid':
    """Finds the closest available neighbor to a physical qubit 'source' on the device.

        Args:
            source: a physical qubit on the device.

        Returns:
            the closest available physical qubit to 'source'.

        Raises:
            ValueError: if there are no available qubits left on the device.
        """
    for _, successors in nx.bfs_successors(self.device_graph, source):
        for successor in successors:
            if successor not in mapped_physicals:
                return successor
    raise ValueError('No available physical qubits left on the device.')