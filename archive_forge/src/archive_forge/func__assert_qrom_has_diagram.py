import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
def _assert_qrom_has_diagram(qrom: cirq_ft.QROM, expected_diagram: str):
    gh = cirq_ft.testing.GateHelper(qrom)
    op = gh.operation
    context = cirq.DecompositionContext(qubit_manager=cirq.GreedyQubitManager(prefix='anc'))
    circuit = cirq.Circuit(cirq.decompose_once(op, context=context))
    selection = [*itertools.chain.from_iterable((gh.quregs[reg.name] for reg in qrom.selection_registers))]
    selection = [q for q in selection if q in circuit.all_qubits()]
    anc = sorted(set(circuit.all_qubits()) - set(op.qubits))
    selection_and_anc = (selection[0],) + sum(zip(selection[1:], anc), ())
    qubit_order = cirq.QubitOrder.explicit(selection_and_anc, fallback=cirq.QubitOrder.DEFAULT)
    cirq.testing.assert_has_diagram(circuit, expected_diagram, qubit_order=qubit_order)