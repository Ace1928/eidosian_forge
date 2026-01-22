from typing import (
import numpy as np
from cirq import protocols, _compat
from cirq.circuits import AbstractCircuit, Alignment, Circuit
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.type_workarounds import NotImplementedType
@classmethod
def _from_moments(cls, moments: Iterable['cirq.Moment']) -> 'FrozenCircuit':
    new_circuit = FrozenCircuit()
    new_circuit._moments = tuple(moments)
    return new_circuit