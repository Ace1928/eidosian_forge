import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
class TestOp(cirq.Operation):

    def __init__(self, *, has_unitary: bool):
        self.count = 0
        self.has_unitary = has_unitary

    def _act_on_(self, sim_state):
        self.count += 1
        return True

    def with_qubits(self, qubits):
        pass

    @property
    def qubits(self):
        return (q,)

    def _has_unitary_(self):
        return self.has_unitary