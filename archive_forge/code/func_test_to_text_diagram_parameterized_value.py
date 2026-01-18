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
def test_to_text_diagram_parameterized_value(circuit_cls):
    q = cirq.NamedQubit('cube')

    class PGate(cirq.testing.SingleQubitGate):

        def __init__(self, val):
            self.val = val

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
            return cirq.CircuitDiagramInfo(('P',), self.val)
    c = circuit_cls(PGate(1).on(q), PGate(2).on(q), PGate(sympy.Symbol('a')).on(q), PGate(sympy.Symbol('%$&#*(')).on(q))
    assert str(c).strip() == 'cube: ───P───P^2───P^a───P^(%$&#*()───'