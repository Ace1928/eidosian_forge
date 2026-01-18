from typing import List, Sequence, Tuple
import itertools
import numpy as np
import pytest
import sympy
import cirq
def test_parameterized_gates():
    t = sympy.Symbol('t')
    with pytest.raises(ValueError):
        cphase_gate = cirq.CZPowGate(exponent=t)
        fsim_gate = FakeSycamoreGate()
        cirq.decompose_cphase_into_two_fsim(cphase_gate, fsim_gate=fsim_gate)
    with pytest.raises(ValueError):
        cphase_gate = cirq.CZ
        fsim_gate = cirq.FSimGate(theta=t, phi=np.pi / 2)
        cirq.decompose_cphase_into_two_fsim(cphase_gate, fsim_gate=fsim_gate)
    with pytest.raises(ValueError):
        cphase_gate = cirq.CZ
        fsim_gate = cirq.FSimGate(theta=np.pi / 2, phi=t)
        cirq.decompose_cphase_into_two_fsim(cphase_gate, fsim_gate=fsim_gate)