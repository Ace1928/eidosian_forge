from typing import (
import numpy as np
from cirq import protocols, _compat
from cirq.circuits import AbstractCircuit, Alignment, Circuit
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.type_workarounds import NotImplementedType
def to_op(self) -> 'cirq.CircuitOperation':
    """Creates a CircuitOperation wrapping this circuit."""
    from cirq.circuits import CircuitOperation
    return CircuitOperation(self)