import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('data,num_controls', [pytest.param(data, num_controls, id=f'{num_controls}-data{idx}', marks=pytest.mark.slow if num_controls == 2 and idx == 2 else ()) for idx, data in enumerate([[[1, 2, 3, 4, 5]], [[1, 2, 3], [4, 5, 10]], [[1], [2], [3], [4], [5], [6]]]) for num_controls in [0, 1, 2]])
@allow_deprecated_cirq_ft_use_in_tests
def test_qrom_1d(data, num_controls):
    qrom = cirq_ft.QROM.build(*data, num_controls=num_controls)
    greedy_mm = cirq.GreedyQubitManager('a', maximize_reuse=True)
    g = cirq_ft.testing.GateHelper(qrom, context=cirq.DecompositionContext(greedy_mm))
    decomposed_circuit = cirq.Circuit(cirq.decompose(g.operation, context=g.context))
    inverse = cirq.Circuit(cirq.decompose(g.operation ** (-1), context=g.context))
    assert len(inverse.all_qubits()) <= infra.total_bits(g.r) + g.r.get_left('selection').total_bits() + num_controls
    assert inverse.all_qubits() == decomposed_circuit.all_qubits()
    for selection_integer in range(len(data[0])):
        for cval in range(2):
            qubit_vals = {x: 0 for x in g.all_qubits}
            qubit_vals.update(zip(g.quregs['selection'], iter_bits(selection_integer, g.r.get_left('selection').total_bits())))
            if num_controls:
                qubit_vals.update(zip(g.quregs['control'], [cval] * num_controls))
            initial_state = [qubit_vals[x] for x in g.all_qubits]
            if cval or not num_controls:
                for ti, d in enumerate(data):
                    target = g.quregs[f'target{ti}']
                    qubit_vals.update(zip(target, iter_bits(d[selection_integer], len(target))))
            final_state = [qubit_vals[x] for x in g.all_qubits]
            cirq_ft.testing.assert_circuit_inp_out_cirqsim(decomposed_circuit, g.all_qubits, initial_state, final_state)
            cirq_ft.testing.assert_circuit_inp_out_cirqsim(decomposed_circuit + inverse, g.all_qubits, initial_state, initial_state)
            cirq_ft.testing.assert_circuit_inp_out_cirqsim(decomposed_circuit + inverse, g.all_qubits, final_state, final_state)