from itertools import combinations
from string import ascii_lowercase
from typing import Sequence, Dict, Tuple
import numpy as np
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
class ExampleGate(cirq.Gate):

    def __init__(self, wire_symbols: Sequence[str]) -> None:
        self._num_qubits = len(wire_symbols)
        self._wire_symbols = tuple(wire_symbols)

    def num_qubits(self) -> int:
        return self._num_qubits

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs):
        return self._wire_symbols