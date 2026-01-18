import numpy as np
import pytest
import sympy
import cirq
def test_phased_fsim_json_dict():
    assert cirq.PhasedFSimGate(theta=0.12, zeta=0.34, chi=0.56, gamma=0.78, phi=0.9)._json_dict_() == {'theta': 0.12, 'zeta': 0.34, 'chi': 0.56, 'gamma': 0.78, 'phi': 0.9}