import functools
import itertools
from typing import Dict, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING
from cirq import ops
from cirq.contrib.acquaintance.gates import acquaint
from cirq.contrib.acquaintance.permutation import PermutationGate
from cirq.contrib.acquaintance.shift import CircularShiftGate
A swap network that generalizes the circular shift gate.

    Given a specification of two partitions, implements a swap network that has
    the overall effect of:
        * For every pair of parts, one from each partition, acquainting the
            union of the corresponding qubits.
        * Circularly shifting the two sets of qubits.

    Args:
        left_part_lens: The sizes of the parts in the partition of the first
            set of qubits.
        right_part_lens: The sizes of the parts in the partition of the second
            set of qubits.
        swap_gate: The gate to use when decomposing.

    Attributes:
        part_lens: A mapping from the side (as a str, 'left' or 'right') to the
            part sizes of the corresponding partition.
        swap_gate: The gate to use when decomposing.
    