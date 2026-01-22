from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
class QubitGate(cirq.Gate):

    def _num_qubits_(self):
        return 2

    def _qid_shape_(self):
        return NotImplemented