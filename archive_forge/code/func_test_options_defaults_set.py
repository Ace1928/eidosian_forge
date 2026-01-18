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
def test_options_defaults_set():
    o1 = XEBPhasedFSimCharacterizationOptions(characterize_zeta=True, characterize_chi=True, characterize_gamma=True, characterize_theta=False, characterize_phi=False)
    assert o1.defaults_set() is False
    with pytest.raises(ValueError):
        o1.get_initial_simplex_and_names()
    o2 = XEBPhasedFSimCharacterizationOptions(characterize_zeta=True, characterize_chi=True, characterize_gamma=True, characterize_theta=False, characterize_phi=False, zeta_default=0.1, chi_default=0.2, gamma_default=0.3)
    with pytest.raises(ValueError):
        _ = o2.defaults_set()
    o3 = XEBPhasedFSimCharacterizationOptions(characterize_zeta=True, characterize_chi=True, characterize_gamma=True, characterize_theta=False, characterize_phi=False, zeta_default=0.1, chi_default=0.2, gamma_default=0.3, theta_default=0.0, phi_default=0.0)
    assert o3.defaults_set() is True