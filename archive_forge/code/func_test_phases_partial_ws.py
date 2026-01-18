from typing import cast, Iterable
import dataclasses
import numpy as np
import pytest
import sympy
import cirq
def test_phases_partial_ws():
    q = cirq.NamedQubit('q')
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    z = sympy.Symbol('z')
    assert_optimizes(before=quick_circuit([cirq.X(q)], [cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(q)]), expected=quick_circuit([cirq.PhasedXPowGate(phase_exponent=-0.25, exponent=0.5).on(q)], [cirq.X(q)]))
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.25).on(q)], [cirq.X(q) ** 0.5]), expected=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.5, exponent=0.5).on(q)], [cirq.PhasedXPowGate(phase_exponent=0.25).on(q)]))
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.25).on(q)], [cirq.PhasedXPowGate(phase_exponent=0.5, exponent=0.75).on(q)]), expected=quick_circuit([cirq.X(q) ** 0.75], [cirq.PhasedXPowGate(phase_exponent=0.25).on(q)]))
    assert_optimizes(before=quick_circuit([cirq.X(q)], [cirq.PhasedXPowGate(exponent=-0.25, phase_exponent=0.5).on(q)]), expected=quick_circuit([cirq.PhasedXPowGate(exponent=-0.25, phase_exponent=-0.5).on(q)], [cirq.X(q)]))
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=x).on(q)], [cirq.PhasedXPowGate(phase_exponent=y, exponent=z).on(q)]), expected=quick_circuit([cirq.PhasedXPowGate(phase_exponent=2 * x - y, exponent=z).on(q)], [cirq.PhasedXPowGate(phase_exponent=x).on(q)]), eject_parameterized=True)