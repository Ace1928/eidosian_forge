import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_text_diagram():
    q0, q1, q2 = _make_qubits(3)
    circuit = cirq.Circuit(cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z})), cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Y})) ** 0.25, cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z, q1: cirq.Z, q2: cirq.Z})), cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z, q1: cirq.Y, q2: cirq.X}, -1)) ** 0.5, cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z, q1: cirq.Y, q2: cirq.X}), exponent_neg=sympy.Symbol('a')), cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z, q1: cirq.Y, q2: cirq.X}, -1), exponent_neg=sympy.Symbol('b')), cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z}), qubits=[q0, q1], exponent_neg=0.5))
    cirq.testing.assert_has_diagram(circuit, '\nq0: ───[Z]───[Y]^0.25───[Z]───[Z]────────[Z]─────[Z]────────[Z]───────\n                        │     │          │       │          │\nq1: ────────────────────[Z]───[Y]────────[Y]─────[Y]────────[I]^0.5───\n                        │     │          │       │\nq2: ────────────────────[Z]───[X]^-0.5───[X]^a───[X]^(-b)─────────────\n')