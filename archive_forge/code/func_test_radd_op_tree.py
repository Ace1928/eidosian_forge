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
def test_radd_op_tree(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = circuit_cls()
    assert [cirq.X(a), cirq.Y(b)] + c == circuit_cls([cirq.Moment([cirq.X(a), cirq.Y(b)])])
    assert cirq.X(a) + c == circuit_cls(cirq.X(a))
    assert [cirq.X(a)] + c == circuit_cls(cirq.X(a))
    assert [[[cirq.X(a)], []]] + c == circuit_cls(cirq.X(a))
    assert (cirq.X(a),) + c == circuit_cls(cirq.X(a))
    assert (cirq.X(a) for _ in range(1)) + c == circuit_cls(cirq.X(a))
    with pytest.raises(AttributeError):
        _ = cirq.X + c
    with pytest.raises(TypeError):
        _ = 0 + c
    if circuit_cls == cirq.FrozenCircuit:
        d = cirq.FrozenCircuit(cirq.Y(b))
    else:
        d = cirq.Circuit()
        d.append(cirq.Y(b))
    assert [cirq.X(a)] + d == circuit_cls([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.Y(b)])])
    assert cirq.Moment([cirq.X(a)]) + d == circuit_cls([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.Y(b)])])