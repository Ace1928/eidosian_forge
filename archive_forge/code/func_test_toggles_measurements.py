from typing import cast, Iterable
import dataclasses
import numpy as np
import pytest
import sympy
import cirq
def test_toggles_measurements():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    x = sympy.Symbol('x')
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.25).on(a)], [cirq.measure(a, b)]), expected=quick_circuit([cirq.measure(a, b, invert_mask=(True,))]))
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.25).on(b)], [cirq.measure(a, b)]), expected=quick_circuit([cirq.measure(a, b, invert_mask=(False, True))]))
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=x).on(b)], [cirq.measure(a, b)]), expected=quick_circuit([cirq.measure(a, b, invert_mask=(False, True))]), eject_parameterized=True)
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.25).on(a)], [cirq.PhasedXPowGate(phase_exponent=0.25).on(b)], [cirq.measure(a, b)]), expected=quick_circuit([cirq.measure(a, b, invert_mask=(True, True))]))
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.25).on(a)], [cirq.measure(a, b, key='t')]), expected=quick_circuit([cirq.measure(a, b, invert_mask=(True,), key='t')]))
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.25).on(a)], [cirq.measure(a, key='m')], [cirq.X(b).with_classical_controls('m')]), expected=quick_circuit([cirq.measure(a, invert_mask=(True,), key='m')], [cirq.X(b).with_classical_controls('m')]), compare_unitaries=False)