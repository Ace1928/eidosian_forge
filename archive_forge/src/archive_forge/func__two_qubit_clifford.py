import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
def _two_qubit_clifford(q_0: 'cirq.Qid', q_1: 'cirq.Qid', idx: int, cliffords: Cliffords) -> Iterator['cirq.OP_TREE']:
    """Generates a two-qubit Clifford gate.

    An integer (idx) from 0 to 11519 is used to generate a two-qubit Clifford
    gate which is constructed with single-qubit X and Y rotations and CZ gates.
    The decomposition of the Cliffords follow those described in the appendix
    of Barends et al., Nature 508, 500 (https://arxiv.org/abs/1402.4848).

    The integer idx is first decomposed into idx_0 (which ranges from 0 to
    23), idx_1 (ranging from 0 to 23) and idx_2 (ranging from 0 to 19). idx_0
    and idx_1 determine the two single-qubit rotations which happen at the
    beginning of all two-qubit Clifford gates. idx_2 determines the
    subsequent gates in the following:

    a) If idx_2 = 0, do nothing so the Clifford is just two single-qubit
    Cliffords (total of 24*24 = 576 possibilities).

    b) If idx_2 = 1, perform a CZ, followed by -Y/2 on q_0 and Y/2 on q_1,
    followed by another CZ, followed by Y/2 on q_0 and -Y/2 on q_1, followed
    by one more CZ and finally a Y/2 on q_1. The Clifford is then a member of
    the SWAP-like class (total of 24*24 = 576 possibilities).

    c) If 2 <= idx_2 <= 10, perform a CZ followed by a member of the S_1
    group on q_0 and a member of the S_1^(Y/2) group on q_1. The Clifford is
    a member of the CNOT-like class (a total of 3*3*24*24 = 5184 possibilities).

    d) If 11 <= idx_2 <= 19, perform a CZ, followed by Y/2 on q_0 and -X/2 on
    q_1, followed by another CZ, and finally a member of the S_1^(Y/2) group on
    q_0 and a member of the S_1^(X/2) group on q_1. The Clifford is a member
    of the iSWAP-like class (a total of 3*3*24*24 = 5184 possibilities).

    Through the above process, all 11520 members of the two-qubit Clifford
    group may be generated.

    Args:
        q_0: The first qubit under test.
        q_1: The second qubit under test.
        idx: An integer from 0 to 11519.
        cliffords: A NamedTuple that contains single-qubit Cliffords from the
            C1, S1, S_1^(X/2) and S_1^(Y/2) groups.
    """
    idx_0, idx_1, idx_2 = _split_two_q_clifford_idx(idx)
    yield _two_qubit_clifford_starters(q_0, q_1, idx_0, idx_1, cliffords)
    yield _two_qubit_clifford_mixers(q_0, q_1, idx_2, cliffords)