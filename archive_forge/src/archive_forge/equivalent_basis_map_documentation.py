from typing import Dict, Optional, Sequence
import numpy as np
import cirq
from cirq import circuits
Ensure equivalence of basis state mapping.

    Args:
        maps: dictionary of test computational basis input states and
            the output computational basis states that they should be mapped to.
            The states are specified using little endian convention, meaning
            that the last bit of a binary literal will refer to the last qubit's
            value.
        circuit: the circuit to be tested
    Raises:
        AssertionError: Raised in case any basis state is mapped to the wrong
            basis state.
    