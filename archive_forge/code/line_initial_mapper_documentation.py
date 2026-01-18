from typing import Deque, Dict, List, Set, Tuple, TYPE_CHECKING
from collections import deque
import networkx as nx
from cirq.transformers.routing import initial_mapper
from cirq import protocols, value
Finds the closest available neighbor to a physical qubit 'source' on the device.

        Args:
            source: a physical qubit on the device.

        Returns:
            the closest available physical qubit to 'source'.

        Raises:
            ValueError: if there are no available qubits left on the device.
        