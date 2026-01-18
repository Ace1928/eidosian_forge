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
def test_reachable_frontier_from(circuit_cls):
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = circuit_cls(cirq.H(a), cirq.CZ(a, b), cirq.H(b), cirq.CZ(b, c), cirq.H(c), cirq.CZ(c, d), cirq.H(d), cirq.CZ(c, d), cirq.H(c), cirq.CZ(b, c), cirq.H(b), cirq.CZ(a, b), cirq.H(a))
    assert circuit_cls().reachable_frontier_from(start_frontier={}) == {}
    assert circuit.reachable_frontier_from(start_frontier={}) == {}
    assert circuit_cls().reachable_frontier_from(start_frontier={a: 5}) == {a: 5}
    assert circuit_cls().reachable_frontier_from(start_frontier={a: -100}) == {a: 0}
    assert circuit.reachable_frontier_from(start_frontier={a: 100}) == {a: 100}
    assert circuit.reachable_frontier_from({a: -1}) == {a: 1}
    assert circuit.reachable_frontier_from({a: 0}) == {a: 1}
    assert circuit.reachable_frontier_from({a: 1}) == {a: 1}
    assert circuit.reachable_frontier_from({a: 2}) == {a: 11}
    assert circuit.reachable_frontier_from({a: 5}) == {a: 11}
    assert circuit.reachable_frontier_from({a: 10}) == {a: 11}
    assert circuit.reachable_frontier_from({a: 11}) == {a: 11}
    assert circuit.reachable_frontier_from({a: 12}) == {a: 13}
    assert circuit.reachable_frontier_from({a: 13}) == {a: 13}
    assert circuit.reachable_frontier_from({a: 14}) == {a: 14}
    assert circuit.reachable_frontier_from({a: 0, b: 0}) == {a: 11, b: 3}
    assert circuit.reachable_frontier_from({a: 2, b: 2}) == {a: 11, b: 3}
    assert circuit.reachable_frontier_from({a: 0, b: 4}) == {a: 1, b: 9}
    assert circuit.reachable_frontier_from({a: 3, b: 4}) == {a: 11, b: 9}
    assert circuit.reachable_frontier_from({a: 3, b: 9}) == {a: 11, b: 9}
    assert circuit.reachable_frontier_from({a: 3, b: 10}) == {a: 13, b: 13}
    assert circuit.reachable_frontier_from({a: 0, b: 0, c: 0}) == {a: 11, b: 9, c: 5}
    assert circuit.reachable_frontier_from({a: 0, b: 0, c: 0, d: 0}) == {a: 13, b: 13, c: 13, d: 13}
    assert circuit.reachable_frontier_from({a: 0, b: 0, c: 0, d: 0}, is_blocker=lambda op: op == cirq.CZ(b, c)) == {a: 11, b: 3, c: 3, d: 5}