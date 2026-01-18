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
def test_qid_shape_qudit(circuit_cls):

    class PlusOneMod3Gate(cirq.testing.SingleQubitGate):

        def _qid_shape_(self):
            return (3,)

    class C2NotGate(cirq.Gate):

        def _qid_shape_(self):
            return (3, 2)

    class IdentityGate(cirq.testing.SingleQubitGate):

        def _qid_shape_(self):
            return (1,)
    a, b, c = cirq.LineQid.for_qid_shape((3, 2, 1))
    circuit = circuit_cls(PlusOneMod3Gate().on(a), C2NotGate().on(a, b), IdentityGate().on_each(c))
    assert cirq.num_qubits(circuit) == 3
    assert cirq.qid_shape(circuit) == (3, 2, 1)
    assert circuit.qid_shape() == (3, 2, 1)
    assert circuit.qid_shape()
    with pytest.raises(ValueError, match='extra qubits'):
        _ = circuit.qid_shape(qubit_order=[b, c])