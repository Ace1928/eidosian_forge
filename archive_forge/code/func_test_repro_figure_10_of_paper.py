import numpy as np
import pytest
import cirq
import cirq.contrib.bayesian_network as ccb
def test_repro_figure_10_of_paper():
    ir = cirq.NamedQubit('q4_IR')
    oi = cirq.NamedQubit('q3_OI')
    sm = cirq.NamedQubit('q2_SM')
    sp = cirq.NamedQubit('q0_SP')
    gate = ccb.BayesianNetworkGate([('ir', 0.25), ('oi', 0.4), ('sm', None), ('sp', None)], [('sm', ('ir',), [0.7, 0.2]), ('sp', ('sm', 'oi'), [0.1, 0.5, 0.6, 0.8])])
    qubits = [sp, sm, oi, ir]
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits)))
    result = cirq.Simulator().simulate(circuit, qubit_order=qubits, initial_state=0)
    probs = np.asarray([abs(x) ** 2 for x in result.state_vector(copy=True)]).reshape(2, 2, 2, 2)
    np.testing.assert_almost_equal(np.sum(probs[0, :, :, :]), 0.75, decimal=6)
    np.testing.assert_almost_equal(np.sum(probs[:, :, 0, :]), 0.425, decimal=6)
    np.testing.assert_almost_equal(np.sum(probs[:, 0, :, :]), 0.6, decimal=6)
    np.testing.assert_almost_equal(np.sum(probs[:, :, :, 0]), 0.4985, decimal=6)