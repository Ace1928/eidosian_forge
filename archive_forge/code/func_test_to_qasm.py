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
def test_to_qasm(circuit_cls):
    q0 = cirq.NamedQubit('q0')
    circuit = circuit_cls(cirq.X(q0))
    assert circuit.to_qasm() == cirq.qasm(circuit)
    assert circuit.to_qasm() == f'// Generated from Cirq v{cirq.__version__}\n\nOPENQASM 2.0;\ninclude "qelib1.inc";\n\n\n// Qubits: [q0]\nqreg q[1];\n\n\nx q[0];\n'