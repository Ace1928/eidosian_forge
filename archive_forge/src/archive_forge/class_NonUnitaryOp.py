import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
class NonUnitaryOp(cirq.Operation):
    count = 0

    def _act_on_(self, sim_state):
        self.count += 1
        return True

    def with_qubits(self, qubits):
        pass

    @property
    def qubits(self):
        return (q,)