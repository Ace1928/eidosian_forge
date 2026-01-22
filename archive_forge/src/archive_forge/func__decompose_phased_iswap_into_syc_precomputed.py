from typing import Iterator, List, Optional
import itertools
import math
import numpy as np
import cirq
from cirq_google import ops
def _decompose_phased_iswap_into_syc_precomputed(theta: float, a: cirq.Qid, b: cirq.Qid) -> cirq.OP_TREE:
    """Decomposes `cirq.PhasedISwapPowGate` into Sycamore gates using precomputed coefficients.

    This should only be called if the Gate has a phase_exponent of .25. If the gate has an
    exponent of 1, _decompose_phased_iswap_into_syc should be used instead. Converting PhasedISwap
    gates to Sycamore is not supported if neither of these constraints are satisfied.

    This synthesize a PhasedISwap in terms of four sycamore gates.  This compilation converts the
    gate into a circuit involving two CZ gates, which themselves are each represented as two
    Sycamore gates and single-qubit rotations

    Args:
        theta: Rotation parameter for the phased ISWAP.
        a: First qubit to operate on.
        b: Second qubit to operate on.

    Yields:
        A `cirq.OP_TREE` implementing the `cirq.PhasedISwapPowGate` gate using Sycamore gates.
    """
    yield cirq.PhasedXPowGate(phase_exponent=0.41175161497166024, exponent=0.5653807577895922).on(a)
    yield (cirq.PhasedXPowGate(phase_exponent=1.0, exponent=0.5).on(b),)
    yield ((cirq.Z ** 0.7099892314883478).on(b),)
    yield ((cirq.Z ** 0.6746023442550453).on(a),)
    yield ops.SYC(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.5154334589432878, exponent=0.5228733015013345).on(b)
    yield cirq.PhasedXPowGate(phase_exponent=0.06774925307475355).on(a)
    yield (ops.SYC(a, b),)
    yield cirq.PhasedXPowGate(phase_exponent=-0.5987667922766213, exponent=0.4136540654256824).on(a)
    yield (cirq.Z ** (-0.9255092746611595)).on(b)
    yield (cirq.Z ** (-1.333333333333333)).on(a)
    yield cirq.rx(-theta).on(a)
    yield cirq.rx(-theta).on(b)
    yield cirq.PhasedXPowGate(phase_exponent=0.5678998743900456, exponent=0.5863459345743176).on(a)
    yield cirq.PhasedXPowGate(phase_exponent=0.3549946157441739).on(b)
    yield ops.SYC(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.5154334589432878, exponent=0.5228733015013345).on(b)
    yield cirq.PhasedXPowGate(phase_exponent=0.06774925307475355).on(a)
    yield ops.SYC(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.8151665352515929, exponent=0.8906746535691492).on(a)
    yield cirq.PhasedXPowGate(phase_exponent=-0.07449072533884049, exponent=0.5).on(b)
    yield (cirq.Z ** (-0.9255092746611595)).on(b)
    yield (cirq.Z ** (-0.9777346353961884)).on(a)