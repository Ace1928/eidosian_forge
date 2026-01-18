from typing import AbstractSet, Union, Any, Optional, Tuple, TYPE_CHECKING, Dict
import numpy as np
from cirq import protocols, value
from cirq.ops import raw_types
from cirq.type_workarounds import NotImplementedType
def with_num_copies(self, num_copies: int) -> 'ParallelGate':
    """ParallelGate with same sub_gate but different num_copies"""
    return ParallelGate(self.sub_gate, num_copies)