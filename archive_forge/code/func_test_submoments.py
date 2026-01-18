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
def test_submoments(circuit_cls):
    a, b, c, d, e, f = cirq.LineQubit.range(6)
    circuit = circuit_cls(cirq.H.on(a), cirq.H.on(d), cirq.CZ.on(a, d), cirq.CZ.on(b, c), (cirq.CNOT ** 0.5).on(a, d), (cirq.CNOT ** 0.5).on(b, e), (cirq.CNOT ** 0.5).on(c, f), cirq.H.on(c), cirq.H.on(e))
    cirq.testing.assert_has_diagram(circuit, '\n          ┌───────────┐   ┌──────┐\n0: ───H────@───────────────@─────────\n           │               │\n1: ───@────┼@──────────────┼─────────\n      │    ││              │\n2: ───@────┼┼────@─────────┼────H────\n           ││    │         │\n3: ───H────@┼────┼─────────X^0.5─────\n            │    │\n4: ─────────X^0.5┼─────────H─────────\n                 │\n5: ──────────────X^0.5───────────────\n          └───────────┘   └──────┘\n')
    cirq.testing.assert_has_diagram(circuit, '\n  0 1 2 3     4     5\n  │ │ │ │     │     │\n  H @─@ H     │     │\n  │ │ │ │     │     │\n┌╴│ │ │ │     │     │    ╶┐\n│ @─┼─┼─@     │     │     │\n│ │ @─┼─┼─────X^0.5 │     │\n│ │ │ @─┼─────┼─────X^0.5 │\n└╴│ │ │ │     │     │    ╶┘\n  │ │ │ │     │     │\n┌╴│ │ │ │     │     │    ╶┐\n│ @─┼─┼─X^0.5 H     │     │\n│ │ │ H │     │     │     │\n└╴│ │ │ │     │     │    ╶┘\n  │ │ │ │     │     │\n', transpose=True)
    cirq.testing.assert_has_diagram(circuit, '\n          /-----------\\   /------\\\n0: ---H----@---------------@---------\n           |               |\n1: ---@----|@--------------|---------\n      |    ||              |\n2: ---@----||----@---------|----H----\n           ||    |         |\n3: ---H----@|----|---------X^0.5-----\n            |    |\n4: ---------X^0.5|---------H---------\n                 |\n5: --------------X^0.5---------------\n          \\-----------/   \\------/\n', use_unicode_characters=False)
    cirq.testing.assert_has_diagram(circuit, '\n  0 1 2 3     4     5\n  | | | |     |     |\n  H @-@ H     |     |\n  | | | |     |     |\n/ | | | |     |     |     \\\n| @-----@     |     |     |\n| | @---------X^0.5 |     |\n| | | @-------------X^0.5 |\n\\ | | | |     |     |     /\n  | | | |     |     |\n/ | | | |     |     |     \\\n| @-----X^0.5 H     |     |\n| | | H |     |     |     |\n\\ | | | |     |     |     /\n  | | | |     |     |\n', use_unicode_characters=False, transpose=True)