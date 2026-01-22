import dataclasses
import pytest
import numpy as np
import sympy
import cirq
from cirq.transformers.eject_z import _is_swaplike
class UnknownOp(cirq.Operation):

    @property
    def qubits(self):
        return [q]

    def with_qubits(self, *new_qubits):
        raise NotImplementedError()