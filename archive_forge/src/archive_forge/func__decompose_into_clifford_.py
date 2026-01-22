import re
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, value
from cirq.ops import raw_types, gate_features, control_values as cv
from cirq.type_workarounds import NotImplementedType
def _decompose_into_clifford_(self):
    sub = getattr(self.gate, '_decompose_into_clifford_with_qubits_', None)
    if sub is None:
        return NotImplemented
    return sub(self.qubits)