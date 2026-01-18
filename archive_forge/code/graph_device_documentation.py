import abc
import itertools
from typing import Iterable, Optional, TYPE_CHECKING, Tuple, cast
from cirq import devices, ops, value
from cirq.contrib.graph_device.hypergraph import UndirectedHypergraph
Inits UndirectedGraphDevice.

        Args:
            device_graph: An undirected hypergraph whose vertices correspond to
                qubits and whose edges determine allowable operations and their
                durations.
            crosstalk_graph: An undirected hypergraph whose vertices are edges
                of device_graph and whose edges give simultaneity constraints
                thereon.

        Raises:
            TypeError: If the crosstalk graph is not a valid crosstalk graph.
        