from typing import Tuple
import warnings
import numpy as np
import pytest
import cirq
class QuditGate(cirq.Gate):

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (3, 3)

    def _unitary_(self):
        return np.eye(9)

    def _qasm_(self, args: cirq.QasmArgs, qubits: Tuple[cirq.Qid, ...]):
        return NotImplemented