import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_scope_extern_mismatch():
    q = cirq.LineQubit(0)
    inner = cirq.Circuit(cirq.measure(q, key='a'), cirq.X(q).with_classical_controls('b'))
    middle = cirq.Circuit(cirq.measure(q, key=cirq.MeasurementKey('b', ('0',))), cirq.CircuitOperation(inner.freeze(), repetitions=2))
    outer_subcircuit = cirq.CircuitOperation(middle.freeze(), repetitions=2)
    circuit = outer_subcircuit.mapped_circuit(deep=True)
    internal_control_keys = [str(condition) for op in circuit.all_operations() for condition in cirq.control_keys(op)]
    assert internal_control_keys == ['b', 'b', 'b', 'b']
    assert cirq.control_keys(outer_subcircuit) == {cirq.MeasurementKey('b')}
    assert cirq.control_keys(circuit) == {cirq.MeasurementKey('b')}
    cirq.testing.assert_has_diagram(cirq.Circuit(outer_subcircuit), "\n      [                  [ 0: ───M('a')───X─── ]             ]\n      [ 0: ───M('0:b')───[                ║    ]──────────── ]\n0: ───[                  [ b: ════════════^═══ ](loops=2)    ]────────────\n      [                  ║                                   ]\n      [ b: ══════════════╩══════════════════════════════════ ](loops=2)\n      ║\nb: ═══╩═══════════════════════════════════════════════════════════════════\n", use_unicode_characters=True)
    cirq.testing.assert_has_diagram(circuit, "\n0: ───M('0:0:b')───M('0:0:a')───X───M('0:1:a')───X───M('1:0:b')───M('1:0:a')───X───M('1:1:a')───X───\n                                ║                ║                             ║                ║\nb: ═════════════════════════════^════════════════^═════════════════════════════^════════════════^═══\n", use_unicode_characters=True)
    assert circuit == cirq.Circuit(cirq.decompose(outer_subcircuit))