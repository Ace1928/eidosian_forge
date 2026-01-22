from typing import Iterator, List, Optional
import itertools
import math
import numpy as np
import cirq
from cirq_google import ops
def _decompose_swap_into_syc(a: cirq.Qid, b: cirq.Qid):
    """Decomposes `cirq.SWAP` into sycamore gates using precomputed coefficients.

    This should only be called when exponent of `cirq.SwapPowGate` is 1. Other cases are currently
    not supported.

    Args:
        a: First qubit to operate on.
        b: Second qubit to operate on.

    Yields:
        A `cirq.OP_TREE` implementing the `cirq.SWAP` gate using Sycamore gates.
    """
    yield cirq.PhasedXPowGate(phase_exponent=0.44650378384076217, exponent=0.8817921214052824).on(a)
    yield cirq.PhasedXPowGate(phase_exponent=-0.7656774060816165, exponent=0.6628666504604785).on(b)
    yield ops.SYC(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.6277589946716742, exponent=0.5659160932099687).on(a)
    yield ops.SYC(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=0.28890767199499257, exponent=0.4340839067900317).on(b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.22592784059288928).on(a)
    yield ops.SYC(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.4691261557936808, exponent=0.7728525693920243).on(a)
    yield cirq.PhasedXPowGate(phase_exponent=-0.8150261316932077, exponent=0.11820787859471782).on(b)
    yield (cirq.Z ** (-0.7384700844660306)).on(b)
    yield (cirq.Z ** (-0.7034535141382525)).on(a)