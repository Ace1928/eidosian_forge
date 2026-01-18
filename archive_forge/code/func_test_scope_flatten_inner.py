import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_scope_flatten_inner():
    q = cirq.LineQubit(0)
    inner = cirq.Circuit(cirq.measure(q, key='a'), cirq.X(q).with_classical_controls('a'))
    middle = cirq.Circuit(cirq.CircuitOperation(inner.freeze(), repetitions=2, use_repetition_ids=False))
    outer_subcircuit = cirq.CircuitOperation(middle.freeze(), repetitions=2)
    circuit = outer_subcircuit.mapped_circuit(deep=True)
    internal_control_keys = [str(condition) for op in circuit.all_operations() for condition in cirq.control_keys(op)]
    assert internal_control_keys == ['0:a', '0:a', '1:a', '1:a']
    assert not cirq.control_keys(outer_subcircuit)
    assert not cirq.control_keys(circuit)
    cirq.testing.assert_has_diagram(cirq.Circuit(outer_subcircuit), '\n      [       [ 0: ───M───X─── ]                         ]\n0: ───[ 0: ───[       ║   ║    ]──────────────────────── ]────────────\n      [       [ a: ═══@═══^═══ ](loops=2, no_rep_ids)    ](loops=2)\n', use_unicode_characters=True)
    cirq.testing.assert_has_diagram(circuit, '\n0: ─────M───X───M───X───M───X───M───X───\n        ║   ║   ║   ║   ║   ║   ║   ║\n0:a: ═══@═══^═══@═══^═══╬═══╬═══╬═══╬═══\n                        ║   ║   ║   ║\n1:a: ═══════════════════@═══^═══@═══^═══\n', use_unicode_characters=True)