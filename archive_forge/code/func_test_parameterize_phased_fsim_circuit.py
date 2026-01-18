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
@pytest.mark.parametrize('gate', [cirq.SQRT_ISWAP, cirq.FSimGate(np.pi / 4, 0)])
def test_parameterize_phased_fsim_circuit(gate):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = rqcg.random_rotations_between_two_qubit_circuit(q0, q1, depth=3, two_qubit_op_factory=lambda a, b, _: gate(a, b), seed=52)
    p_circuit = parameterize_circuit(circuit, SqrtISwapXEBOptions())
    cirq.testing.assert_has_diagram(p_circuit, '0                                    1\n│                                    │\nY^0.5                                X^0.5\n│                                    │\nPhFSim(theta, zeta, chi, gamma, phi)─PhFSim(theta, zeta, chi, gamma, phi)\n│                                    │\nPhX(0.25)^0.5                        Y^0.5\n│                                    │\nPhFSim(theta, zeta, chi, gamma, phi)─PhFSim(theta, zeta, chi, gamma, phi)\n│                                    │\nY^0.5                                X^0.5\n│                                    │\nPhFSim(theta, zeta, chi, gamma, phi)─PhFSim(theta, zeta, chi, gamma, phi)\n│                                    │\nX^0.5                                PhX(0.25)^0.5\n│                                    │\n    ', transpose=True)