from typing import Iterator, List, Optional
import itertools
import math
import numpy as np
import cirq
from cirq_google import ops
def _swap_rzz(theta: float, q0: cirq.Qid, q1: cirq.Qid) -> cirq.OP_TREE:
    """An implementation of SWAP * exp(-1j * theta * ZZ) using three sycamore gates.

    This builds off of the _rzz method.

    Args:
        theta: The rotation parameter of Rzz Ising coupling gate.
        q0: First qubit to operate on.
        q1: Second qubit to operate on.

    Yields:
        The `cirq.OP_TREE`` that implements ZZ followed by a swap.
    """
    angle_offset = np.pi / 24 - np.pi / 4
    circuit = cirq.Circuit(ops.SYC(q0, q1), _rzz(theta - angle_offset, q0, q1))
    intended_circuit = cirq.Circuit(cirq.SWAP(q0, q1), cirq.ZZPowGate(exponent=2 * theta / np.pi, global_shift=-0.5).on(q0, q1))
    yield _create_corrected_circuit(cirq.unitary(intended_circuit), circuit, q0, q1)