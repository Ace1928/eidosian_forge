import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
def _two_qubit_clifford_starters(q_0: 'cirq.Qid', q_1: 'cirq.Qid', idx_0: int, idx_1: int, cliffords: Cliffords) -> Iterator['cirq.OP_TREE']:
    """Fulfills part (a) for two-qubit Cliffords."""
    c1 = cliffords.c1_in_xy
    yield _single_qubit_gates(c1[idx_0], q_0)
    yield _single_qubit_gates(c1[idx_1], q_1)