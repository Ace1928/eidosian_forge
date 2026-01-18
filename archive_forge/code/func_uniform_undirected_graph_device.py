from typing import Any, Dict, Hashable, Iterable, Mapping, Optional
from cirq import devices, ops
from cirq.contrib.graph_device.graph_device import UndirectedGraphDevice, UndirectedGraphDeviceEdge
from cirq.contrib.graph_device.hypergraph import UndirectedHypergraph
def uniform_undirected_graph_device(edges: Iterable[Iterable[ops.Qid]], edge_label: Optional[UndirectedGraphDeviceEdge]=None) -> UndirectedGraphDevice:
    """An undirected graph device all of whose edges are the same.

    Args:
        edges: The edges.
        edge_label: The label to apply to all edges. Defaults to None.
    """
    labelled_edges: Dict[Iterable[Hashable], Any] = {frozenset(edge): edge_label for edge in edges}
    device_graph = UndirectedHypergraph(labelled_edges=labelled_edges)
    return UndirectedGraphDevice(device_graph=device_graph)