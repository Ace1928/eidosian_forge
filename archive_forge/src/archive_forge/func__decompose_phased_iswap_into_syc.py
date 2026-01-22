from typing import Iterator, List, Optional
import itertools
import math
import numpy as np
import cirq
from cirq_google import ops
def _decompose_phased_iswap_into_syc(phase_exponent: float, a: cirq.Qid, b: cirq.Qid) -> cirq.OP_TREE:
    """Decomposes `cirq.PhasedISwapPowGate` with an exponent of 1 into Sycamore gates.

    This should only be called if the gate has an exponent of 1. Otherwise,
    `_decompose_phased_iswap_into_syc_precomputed` should be used instead. The advantage of using
    this function is that the resulting circuit will be smaller.

    Args:
        phase_exponent: The exponent on the Z gates.
        a: First qubit to operate on.
        b: Second qubit to operate on.

    Yields:
        A `cirq.OP_TREE` implementing the `cirq.PhasedISwapPowGate` gate using Sycamore gates.
    """
    yield (cirq.Z(a) ** phase_exponent,)
    yield (cirq.Z(b) ** (-phase_exponent),)
    yield (_decompose_iswap_into_syc(a, b),)
    yield (cirq.Z(a) ** (-phase_exponent),)
    yield (cirq.Z(b) ** phase_exponent,)