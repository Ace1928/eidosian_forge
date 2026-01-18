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
def test_to_text_diagram_many_qubits_gate_but_multiple_wire_symbols(circuit_cls):

    class BadGate(cirq.testing.ThreeQubitGate):

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> Tuple[str, str]:
            return ('a', 'a')
    q1 = cirq.NamedQubit('(0, 0)')
    q2 = cirq.NamedQubit('(0, 1)')
    q3 = cirq.NamedQubit('(0, 2)')
    c = circuit_cls([cirq.Moment([BadGate().on(q1, q2, q3)])])
    with pytest.raises(ValueError, match='BadGate'):
        c.to_text_diagram()