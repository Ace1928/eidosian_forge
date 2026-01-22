import functools
import itertools
import math
import operator
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, TYPE_CHECKING
from cirq import ops, protocols, value
from cirq.contrib.acquaintance.shift import CircularShiftGate
from cirq.contrib.acquaintance.permutation import (
def _get_max_reach(size: int, round_up: bool=True) -> int:
    if round_up:
        return int(math.ceil(size / 2)) - 1
    return max(size // 2 - 1, 0)