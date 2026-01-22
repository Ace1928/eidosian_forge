from typing import (
import numpy as np
from cirq import protocols, _compat
from cirq.circuits import AbstractCircuit, Alignment, Circuit
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.type_workarounds import NotImplementedType
def _repr_args(self) -> str:
    moments_repr = super()._repr_args()
    tag_repr = ','.join((_compat.proper_repr(t) for t in self._tags))
    return f'{moments_repr}, tags=[{tag_repr}]' if self.tags else moments_repr