import itertools
from typing import Sequence, Tuple
import cirq
import cirq_ft
import pytest
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('selection_bitsize, target_bitsize, control_bitsize', [(3, 5, 1), (2, 4, 2), (1, 2, 3)])
@allow_deprecated_cirq_ft_use_in_tests
def test_unary_iteration_gate(selection_bitsize, target_bitsize, control_bitsize):
    greedy_mm = cirq.GreedyQubitManager(prefix='_a', maximize_reuse=True)
    gate = ApplyXToLthQubit(selection_bitsize, target_bitsize, control_bitsize)
    g = cirq_ft.testing.GateHelper(gate, context=cirq.DecompositionContext(greedy_mm))
    assert len(g.all_qubits) <= 2 * (selection_bitsize + control_bitsize) + target_bitsize - 1
    for n in range(target_bitsize):
        qubit_vals = {q: 0 for q in g.operation.qubits}
        qubit_vals.update({c: 1 for c in g.quregs['control']})
        qubit_vals.update(zip(g.quregs['selection'], iter_bits(n, selection_bitsize)))
        initial_state = [qubit_vals[x] for x in g.operation.qubits]
        qubit_vals[g.quregs['target'][-(n + 1)]] = 1
        final_state = [qubit_vals[x] for x in g.operation.qubits]
        cirq_ft.testing.assert_circuit_inp_out_cirqsim(g.circuit, g.operation.qubits, initial_state, final_state)