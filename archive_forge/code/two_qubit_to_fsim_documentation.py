from typing import Sequence, Union, List, Iterator, TYPE_CHECKING, Iterable, Optional
import numpy as np
from cirq import circuits, devices, linalg, ops, protocols
Adds single qubit operations to complete a desired interaction.

    Args:
        desired: The kak decomposition of the desired operation.
        qubits: The pair of qubits that is being operated on.
        operations: A list of operations that composes into the desired kak
            interaction coefficients, but may not have the desired before/after
            single qubit operations or the desired global phase.

    Returns:
        A list of operations whose kak decomposition approximately equals the
        desired kak decomposition.
    