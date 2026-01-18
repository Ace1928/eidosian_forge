import multiprocessing
from typing import Dict, Any, Optional
from typing import Sequence
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def test_simulate_circuit_length_validation():
    q0, q1 = cirq.LineQubit.range(2)
    circuits = [rqcg.random_rotations_between_two_qubit_circuit(q0, q1, depth=10, two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b)) for _ in range(2)]
    cycle_depths = np.arange(3, 50, 9, dtype=np.int64)
    with pytest.raises(ValueError, match='.*not long enough.*'):
        _ = simulate_2q_xeb_circuits(circuits=circuits, cycle_depths=cycle_depths)