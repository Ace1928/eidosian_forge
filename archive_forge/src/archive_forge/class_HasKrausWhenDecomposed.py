from typing import Iterable, List, Sequence, Tuple
import numpy as np
import pytest
import cirq
class HasKrausWhenDecomposed(cirq.testing.SingleQubitGate):

    def __init__(self, decomposed_cls):
        self.decomposed_cls = decomposed_cls

    def _decompose_(self, qubits):
        return [self.decomposed_cls().on(q) for q in qubits]