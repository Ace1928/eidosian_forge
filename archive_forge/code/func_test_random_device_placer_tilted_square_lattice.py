import itertools
import pytest
import cirq
import cirq_google as cg
import numpy as np
def test_random_device_placer_tilted_square_lattice():
    topo = cirq.TiltedSquareLattice(4, 2)
    qubits = sorted(topo.nodes_to_gridqubits().values())
    circuit = cirq.experiments.random_rotations_between_grid_interaction_layers_circuit(qubits, depth=8, two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b))
    assert not all((q in cg.Sycamore23.metadata.qubit_set for q in circuit.all_qubits()))
    qp = cg.RandomDevicePlacer()
    circuit2, mapping = qp.place_circuit(circuit, problem_topology=topo, shared_rt_info=cg.SharedRuntimeInfo(run_id='1', device=cg.Sycamore23), rs=np.random.RandomState(1))
    assert circuit is not circuit2
    assert circuit != circuit2
    assert all((q in cg.Sycamore23.metadata.qubit_set for q in circuit2.all_qubits()))
    for k, v in mapping.items():
        assert k != v