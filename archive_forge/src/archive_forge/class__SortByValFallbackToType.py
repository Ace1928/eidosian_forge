import itertools
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, ops, qis, _compat
from cirq._import import LazyLoader
from cirq.ops import raw_types, op_tree
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
class _SortByValFallbackToType:

    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        try:
            return self.value < other.value
        except TypeError:
            t1 = type(self.value)
            t2 = type(other.value)
            return str(t1) < str(t2)