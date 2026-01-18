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
def test_qid_shape_qubit(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    circuit = circuit_cls([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.X(b)])])
    assert cirq.qid_shape(circuit) == (2, 2)
    assert cirq.num_qubits(circuit) == 2
    assert circuit.qid_shape() == (2, 2)
    assert circuit.qid_shape(qubit_order=[c, a, b]) == (2, 2, 2)
    with pytest.raises(ValueError, match='extra qubits'):
        _ = circuit.qid_shape(qubit_order=[a])