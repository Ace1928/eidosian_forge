from typing import Iterator, List, Optional
import itertools
import math
import numpy as np
import cirq
from cirq_google import ops
An implementation of SWAP * exp(-1j * theta * ZZ) using three sycamore gates.

    This builds off of the _rzz method.

    Args:
        theta: The rotation parameter of Rzz Ising coupling gate.
        q0: First qubit to operate on.
        q1: Second qubit to operate on.

    Yields:
        The `cirq.OP_TREE`` that implements ZZ followed by a swap.
    