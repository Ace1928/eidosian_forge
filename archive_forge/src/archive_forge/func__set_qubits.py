import abc
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, value
from cirq.type_workarounds import NotImplementedType
def _set_qubits(self, qubits: Sequence['cirq.Qid']):
    self._qubits = tuple(qubits)
    self._qubit_map = {q: i for i, q in enumerate(self.qubits)}