from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
class NoWireSymbols(cirq.GlobalPhaseGate):

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        return expected