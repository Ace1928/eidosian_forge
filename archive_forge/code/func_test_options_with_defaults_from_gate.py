import itertools
import multiprocessing
from typing import Iterable
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.xeb_fitting import (
from cirq.experiments.xeb_sampling import sample_2q_xeb_circuits
def test_options_with_defaults_from_gate():
    options = XEBPhasedFSimCharacterizationOptions().with_defaults_from_gate(cirq.ISWAP ** 0.5)
    np.testing.assert_allclose(options.theta_default, -np.pi / 4)
    options = XEBPhasedFSimCharacterizationOptions().with_defaults_from_gate(cirq.ISWAP ** (-0.5))
    np.testing.assert_allclose(options.theta_default, np.pi / 4)
    options = XEBPhasedFSimCharacterizationOptions().with_defaults_from_gate(cirq.FSimGate(0.1, 0.2))
    assert options.theta_default == 0.1
    assert options.phi_default == 0.2
    options = XEBPhasedFSimCharacterizationOptions().with_defaults_from_gate(cirq.PhasedFSimGate(0.1))
    assert options.theta_default == 0.1
    assert options.phi_default == 0.0
    assert options.zeta_default == 0.0
    with pytest.raises(ValueError):
        _ = XEBPhasedFSimCharacterizationOptions().with_defaults_from_gate(cirq.CZ)