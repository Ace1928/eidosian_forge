import math
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
from sympy.combinatorics import GrayCode
from cirq import value
from cirq.ops import common_gates, pauli_gates, raw_types
def _qid_shape_(self) -> Tuple[int, ...]:
    return (2,) * len(self._init_probs)