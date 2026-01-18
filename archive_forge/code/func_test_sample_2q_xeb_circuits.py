import glob
import itertools
from typing import Iterable
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.xeb_sampling import sample_2q_xeb_circuits
def test_sample_2q_xeb_circuits():
    q0 = cirq.NamedQubit('a')
    q1 = cirq.NamedQubit('b')
    circuits = [rqcg.random_rotations_between_two_qubit_circuit(q0, q1, depth=20, two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b)) for _ in range(2)]
    cycle_depths = np.arange(3, 20, 6)
    df = sample_2q_xeb_circuits(sampler=cirq.Simulator(), circuits=circuits, cycle_depths=cycle_depths, shuffle=np.random.RandomState(10))
    assert len(df) == len(cycle_depths) * len(circuits)
    for (circuit_i, cycle_depth), row in df.iterrows():
        assert 0 <= circuit_i < len(circuits)
        assert cycle_depth in cycle_depths
        assert len(row['sampled_probs']) == 4
        assert np.isclose(np.sum(row['sampled_probs']), 1)