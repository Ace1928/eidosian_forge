import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('data,num_controls', [pytest.param(data, num_controls, id=f'{num_controls}-data{idx}', marks=pytest.mark.slow if num_controls == 2 and idx == 0 else ()) for idx, data in enumerate([[np.arange(6).reshape(2, 3), 4 * np.arange(6).reshape(2, 3)], [np.arange(8).reshape(2, 2, 2)]]) for num_controls in [0, 1, 2]])
@allow_deprecated_cirq_ft_use_in_tests
def test_qrom_multi_dim(data, num_controls):
    selection_bitsizes = tuple(((s - 1).bit_length() for s in data[0].shape))
    target_bitsizes = tuple((int(np.max(d)).bit_length() for d in data))
    qrom = cirq_ft.QROM(data, selection_bitsizes=selection_bitsizes, target_bitsizes=target_bitsizes, num_controls=num_controls)
    greedy_mm = cirq.GreedyQubitManager('a', maximize_reuse=True)
    g = cirq_ft.testing.GateHelper(qrom, context=cirq.DecompositionContext(greedy_mm))
    decomposed_circuit = cirq.Circuit(cirq.decompose(g.operation, context=g.context))
    inverse = cirq.Circuit(cirq.decompose(g.operation ** (-1), context=g.context))
    assert len(inverse.all_qubits()) <= infra.total_bits(g.r) + infra.total_bits(qrom.selection_registers) + num_controls
    assert inverse.all_qubits() == decomposed_circuit.all_qubits()
    lens = tuple((reg.total_bits() for reg in qrom.selection_registers))
    for idxs in itertools.product(*[range(dim) for dim in data[0].shape]):
        qubit_vals = {x: 0 for x in g.all_qubits}
        for cval in range(2):
            if num_controls:
                qubit_vals.update(zip(g.quregs['control'], [cval] * num_controls))
            for isel in range(len(idxs)):
                qubit_vals.update(zip(g.quregs[f'selection{isel}'], iter_bits(idxs[isel], lens[isel])))
            initial_state = [qubit_vals[x] for x in g.all_qubits]
            if cval or not num_controls:
                for ti, d in enumerate(data):
                    target = g.quregs[f'target{ti}']
                    qubit_vals.update(zip(target, iter_bits(int(d[idxs]), len(target))))
            final_state = [qubit_vals[x] for x in g.all_qubits]
            qubit_vals = {x: 0 for x in g.all_qubits}
            cirq_ft.testing.assert_circuit_inp_out_cirqsim(decomposed_circuit, g.all_qubits, initial_state, final_state)