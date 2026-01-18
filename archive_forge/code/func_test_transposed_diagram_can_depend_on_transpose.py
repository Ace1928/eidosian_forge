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
@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_transposed_diagram_can_depend_on_transpose(circuit_cls):

    class TestGate(cirq.Gate):

        def num_qubits(self):
            return 1

        def _circuit_diagram_info_(self, args):
            return cirq.CircuitDiagramInfo(wire_symbols=('t' if args.transpose else 'r',))
    c = cirq.Circuit(TestGate()(cirq.NamedQubit('a')))
    cirq.testing.assert_has_diagram(c, 'a: ───r───')
    cirq.testing.assert_has_diagram(c, '\na\n│\nt\n│\n', transpose=True)