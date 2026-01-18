import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_unmeasured_condition():
    q0 = cirq.LineQubit(0)
    bad_circuit = cirq.Circuit(cirq.X(q0).with_classical_controls('a'))
    with pytest.raises(ValueError, match='Measurement key a missing when testing classical control'):
        _ = cirq.Simulator().simulate(bad_circuit)