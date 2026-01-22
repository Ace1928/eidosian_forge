import abc
from typing import TYPE_CHECKING, Optional, FrozenSet, Iterable
import networkx as nx
from cirq import value
@value.value_equality
class DeviceMetadata:
    """Parent type for all device specific metadata classes."""

    def __init__(self, qubits: Iterable['cirq.Qid'], nx_graph: 'nx.Graph'):
        """Construct a DeviceMetadata object.

        Args:
            qubits: Iterable of `cirq.Qid`s that exist on the device.
            nx_graph: `nx.Graph` describing qubit connectivity
                on a device. Nodes represent qubits, directed edges indicate
                directional coupling, undirected edges indicate bi-directional
                coupling.
        """
        self._qubits_set: FrozenSet['cirq.Qid'] = frozenset(qubits)
        self._nx_graph = nx_graph

    @property
    def qubit_set(self) -> FrozenSet['cirq.Qid']:
        """Returns the set of qubits on the device.

        Returns:
            Frozenset of qubits on device.
        """
        return self._qubits_set

    @property
    def nx_graph(self) -> 'nx.Graph':
        """Returns a nx.Graph where nodes are qubits and edges are couple-able qubits.

        Returns:
            `nx.Graph` of device connectivity.
        """
        return self._nx_graph

    def _value_equality_values_(self):
        graph_equality = (tuple(sorted(self._nx_graph.nodes())), tuple(sorted(self._nx_graph.edges(data='directed'))))
        return (self._qubits_set, graph_equality)

    def _json_dict_(self):
        graph_payload = nx.readwrite.json_graph.node_link_data(self._nx_graph)
        qubits_payload = sorted(list(self._qubits_set))
        return {'qubits': qubits_payload, 'nx_graph': graph_payload}

    @classmethod
    def _from_json_dict_(cls, qubits: Iterable['cirq.Qid'], nx_graph: 'nx.Graph', **kwargs):
        graph_obj = nx.readwrite.json_graph.node_link_graph(nx_graph)
        return cls(qubits, graph_obj)