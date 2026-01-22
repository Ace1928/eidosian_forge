from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.barrier import Barrier
@staticmethod
def _collect_potential_merges(dag, barriers):
    """Return the potential merges.

        Returns a dict of DAGOpNode: (Barrier, [DAGOpNode]) objects, where the barrier needs to be
        inserted where the corresponding index DAGOpNode appears in the main DAG, in replacement of
        the listed DAGOpNodes.
        """
    if len(barriers) < 2:
        return None
    node_to_barrier_qubits = {}
    current_barrier = barriers[0]
    end_of_barrier = current_barrier
    current_barrier_nodes = [current_barrier]
    current_qubits = set(current_barrier.qargs)
    current_ancestors = dag.ancestors(current_barrier)
    current_descendants = dag.descendants(current_barrier)
    barrier_to_add = Barrier(len(current_qubits))
    for next_barrier in barriers[1:]:
        if dag._multi_graph.has_edge(end_of_barrier._node_id, next_barrier._node_id):
            next_ancestors = {nd for nd in dag.ancestors(next_barrier) if nd not in current_barrier_nodes}
            next_descendants = {nd for nd in dag.descendants(next_barrier) if nd not in current_barrier_nodes}
            next_qubits = set(next_barrier.qargs)
            if not current_qubits.isdisjoint(next_qubits) and current_ancestors.isdisjoint(next_descendants) and current_descendants.isdisjoint(next_ancestors):
                current_ancestors = current_ancestors | next_ancestors
                current_descendants = current_descendants | next_descendants
                current_qubits = current_qubits | next_qubits
                barrier_to_add = Barrier(len(current_qubits))
                end_of_barrier = next_barrier
                current_barrier_nodes.append(end_of_barrier)
                continue
        if barrier_to_add:
            node_to_barrier_qubits[end_of_barrier] = (barrier_to_add, current_barrier_nodes)
        current_qubits = set(next_barrier.qargs)
        current_ancestors = dag.ancestors(next_barrier)
        current_descendants = dag.descendants(next_barrier)
        barrier_to_add = Barrier(len(current_qubits))
        current_barrier_nodes = []
        end_of_barrier = next_barrier
        current_barrier_nodes.append(end_of_barrier)
    if barrier_to_add:
        node_to_barrier_qubits[end_of_barrier] = (barrier_to_add, current_barrier_nodes)
    return node_to_barrier_qubits