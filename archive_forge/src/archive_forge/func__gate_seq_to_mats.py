import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
def _gate_seq_to_mats(gate_seq: Sequence['cirq.Gate']) -> np.ndarray:
    mat_rep = protocols.unitary(gate_seq[0])
    for gate in gate_seq[1:]:
        mat_rep = np.dot(protocols.unitary(gate), mat_rep)
    return mat_rep