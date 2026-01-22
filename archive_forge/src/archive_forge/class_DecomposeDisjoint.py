import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
class DecomposeDisjoint(cirq.Gate):

    def num_qubits(self):
        return 2

    def _decompose_(self, qubits):
        yield cirq.H(qubits[1])