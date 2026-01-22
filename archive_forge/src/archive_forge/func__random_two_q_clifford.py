import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
def _random_two_q_clifford(q_0: 'cirq.Qid', q_1: 'cirq.Qid', num_cfds: int, cfd_matrices: np.ndarray, cliffords: Cliffords) -> 'cirq.Circuit':
    clifford_group_size = 11520
    idx_list = list(np.random.choice(clifford_group_size, num_cfds))
    circuit = circuits.Circuit()
    for idx in idx_list:
        circuit.append(_two_qubit_clifford(q_0, q_1, idx, cliffords))
    inv_idx = _find_inv_matrix(protocols.unitary(circuit), cfd_matrices)
    circuit.append(_two_qubit_clifford(q_0, q_1, inv_idx, cliffords))
    return circuit