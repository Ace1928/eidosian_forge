import itertools
import re
from typing import cast, Tuple, Union
import numpy as np
import pytest
import sympy
import cirq
from cirq import protocols
from cirq.type_workarounds import NotImplementedType
class MockGate(cirq.testing.TwoQubitGate):

    def __init__(self, exponent_qubit_index=None):
        self._exponent_qubit_index = exponent_qubit_index

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs) -> protocols.CircuitDiagramInfo:
        self.captured_diagram_args = args
        return cirq.CircuitDiagramInfo(wire_symbols=tuple(['M1', 'M2']), exponent=1, exponent_qubit_index=self._exponent_qubit_index, connected=True)

    def _has_mixture_(self):
        return True