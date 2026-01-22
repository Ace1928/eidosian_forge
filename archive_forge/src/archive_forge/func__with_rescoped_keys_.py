import re
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, value
from cirq.ops import raw_types, gate_features, control_values as cv
from cirq.type_workarounds import NotImplementedType
def _with_rescoped_keys_(self, path: Tuple[str, ...], bindable_keys: FrozenSet['cirq.MeasurementKey']):
    new_gate = protocols.with_rescoped_keys(self.gate, path, bindable_keys)
    if new_gate is self.gate:
        return self
    return new_gate.on(*self.qubits)