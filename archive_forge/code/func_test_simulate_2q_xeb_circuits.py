import multiprocessing
from typing import Dict, Any, Optional
from typing import Sequence
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def test_simulate_2q_xeb_circuits():
    q0, q1 = cirq.LineQubit.range(2)
    circuits = [rqcg.random_rotations_between_two_qubit_circuit(q0, q1, depth=50, two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b)) for _ in range(2)]
    cycle_depths = np.arange(3, 50, 9)
    df = simulate_2q_xeb_circuits(circuits=circuits, cycle_depths=cycle_depths)
    assert len(df) == len(cycle_depths) * len(circuits)
    for (circuit_i, cycle_depth), row in df.iterrows():
        assert 0 <= circuit_i < len(circuits)
        assert cycle_depth in cycle_depths
        assert len(row['pure_probs']) == 4
        assert np.isclose(np.sum(row['pure_probs']), 1)
    with multiprocessing.Pool() as pool:
        df2 = simulate_2q_xeb_circuits(circuits, cycle_depths, pool=pool)
    pd.testing.assert_frame_equal(df, df2)