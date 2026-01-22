import functools
import itertools
from typing import Dict, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING
from cirq import ops
from cirq.contrib.acquaintance.gates import acquaint
from cirq.contrib.acquaintance.permutation import PermutationGate
from cirq.contrib.acquaintance.shift import CircularShiftGate
def acquaintance_size(self) -> int:
    return sum((max(self.part_lens[side]) for side in ('left', 'right')))