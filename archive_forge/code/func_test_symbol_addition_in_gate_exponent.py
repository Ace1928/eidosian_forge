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
def test_symbol_addition_in_gate_exponent(circuit_cls):
    qubit = cirq.NamedQubit('a')
    circuit = circuit_cls(cirq.X(qubit) ** 0.5, cirq.YPowGate(exponent=sympy.Symbol('a') + sympy.Symbol('b')).on(qubit))
    cirq.testing.assert_has_diagram(circuit, 'a: ───X^0.5───Y^(a + b)───', use_unicode_characters=True)
    cirq.testing.assert_has_diagram(circuit, '\na\n│\nX^0.5\n│\nY^(a + b)\n│\n', use_unicode_characters=True, transpose=True)
    cirq.testing.assert_has_diagram(circuit, 'a: ---X^0.5---Y^(a + b)---', use_unicode_characters=False)
    cirq.testing.assert_has_diagram(circuit, '\na\n|\nX^0.5\n|\nY^(a + b)\n|\n\n ', use_unicode_characters=False, transpose=True)