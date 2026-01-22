import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
def _single_qubit_gates(gate_seq: Sequence['cirq.Gate'], qubit: 'cirq.Qid') -> Iterator['cirq.OP_TREE']:
    for gate in gate_seq:
        yield gate(qubit)