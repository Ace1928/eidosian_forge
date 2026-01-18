import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
def test_zero_target_operations_go_below_diagram():

    class CustomOperationAnnotation(cirq.Operation):

        def __init__(self, text: str):
            self.text = text

        def with_qubits(self, *new_qubits):
            raise NotImplementedError()

        @property
        def qubits(self):
            return ()

        def _circuit_diagram_info_(self, args) -> str:
            return self.text

    class CustomOperationAnnotationNoInfo(cirq.Operation):

        def with_qubits(self, *new_qubits):
            raise NotImplementedError()

        @property
        def qubits(self):
            return ()

        def __str__(self):
            return 'custom!'

    class CustomGateAnnotation(cirq.Gate):

        def __init__(self, text: str):
            self.text = text

        def _num_qubits_(self):
            return 0

        def _circuit_diagram_info_(self, args) -> str:
            return self.text
    cirq.testing.assert_has_diagram(cirq.Circuit(cirq.Moment(CustomOperationAnnotation('a'), CustomGateAnnotation('b').on(), CustomOperationAnnotation('c')), cirq.Moment(CustomOperationAnnotation('e'), CustomOperationAnnotation('d'))), '\n    a   e\n    b   d\n    c\n    ')
    cirq.testing.assert_has_diagram(cirq.Circuit(cirq.Moment(cirq.H(cirq.LineQubit(0)), CustomOperationAnnotation('a'), cirq.global_phase_operation(1j))), '\n0: ─────────────H──────\n\nglobal phase:   0.5π\n                a\n    ')
    cirq.testing.assert_has_diagram(cirq.Circuit(cirq.Moment(cirq.H(cirq.LineQubit(0)), cirq.CircuitOperation(cirq.FrozenCircuit(CustomOperationAnnotation('a'))))), '\n0: ───H───\n      a\n        ')
    cirq.testing.assert_has_diagram(cirq.Circuit(cirq.Moment(cirq.X(cirq.LineQubit(0)), CustomOperationAnnotation('a'), CustomGateAnnotation('b').on(), CustomOperationAnnotation('c')), cirq.Moment(CustomOperationAnnotation('eee'), CustomOperationAnnotation('d')), cirq.Moment(cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(2)), cirq.CNOT(cirq.LineQubit(1), cirq.LineQubit(3)), CustomOperationAnnotationNoInfo(), CustomOperationAnnotation('zzz')), cirq.Moment(cirq.H(cirq.LineQubit(2)))), '\n                ┌────────┐\n0: ───X──────────@───────────────\n                 │\n1: ──────────────┼──────@────────\n                 │      │\n2: ──────────────X──────┼────H───\n                        │\n3: ─────────────────────X────────\n      a   eee    custom!\n      b   d      zzz\n      c\n                └────────┘\n    ')