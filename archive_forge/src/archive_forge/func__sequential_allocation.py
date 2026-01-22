from __future__ import annotations
import typing
import rustworkx as rx
from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import UnassignedReferenceError
def _sequential_allocation(block) -> rx.PyDAG:
    """A helper function to create a DAG of a sequential alignment context."""
    dag = rx.PyDAG()
    edges: list[tuple[int, int]] = []
    prev_id = None
    for elm in block.blocks:
        node_id = dag.add_node(elm)
        if dag.num_nodes() > 1:
            edges.append((prev_id, node_id))
        prev_id = node_id
    dag.add_edges_from_no_data(edges)
    return dag