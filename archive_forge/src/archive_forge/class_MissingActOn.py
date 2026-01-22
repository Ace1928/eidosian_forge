from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
class MissingActOn(cirq.Operation):

    def with_qubits(self, *new_qubits):
        raise NotImplementedError()

    @property
    def qubits(self):
        pass