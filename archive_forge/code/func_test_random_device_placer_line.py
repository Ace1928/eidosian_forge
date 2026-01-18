import itertools
import pytest
import cirq
import cirq_google as cg
import numpy as np
def test_random_device_placer_line():
    topo = cirq.LineTopology(8)
    qubits = cirq.LineQubit.range(8)
    circuit = cirq.testing.random_circuit(qubits, n_moments=8, op_density=1.0, random_state=52)
    qp = cg.RandomDevicePlacer()
    circuit2, mapping = qp.place_circuit(circuit, problem_topology=topo, shared_rt_info=cg.SharedRuntimeInfo(run_id='1', device=cg.Sycamore23), rs=np.random.RandomState(1))
    assert circuit is not circuit2
    assert circuit != circuit2
    assert all((q in cg.Sycamore23.metadata.qubit_set for q in circuit2.all_qubits()))
    for k, v in mapping.items():
        assert k != v