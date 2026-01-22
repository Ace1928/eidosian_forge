from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
class FixedQids(cirq.Operation):

    def with_qubits(self, *new_qids):
        raise NotImplementedError