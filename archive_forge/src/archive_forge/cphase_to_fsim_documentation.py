from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import devices, ops, protocols
Decomposes CZPowGate into two FSimGates.

    This function implements the decomposition described in section VII F I
    of https://arxiv.org/abs/1910.11333.

    The decomposition results in exactly two FSimGates and a few single-qubit
    rotations. It is feasible if and only if one of the following conditions
    is met:

        |sin(θ)| <= |sin(δ/4)| <= |sin(φ/2)|
        |sin(φ/2)| <= |sin(δ/4)| <= |sin(θ)|

    where:

         θ = fsim_gate.theta,
         φ = fsim_gate.phi,
         δ = -π * cphase_gate.exponent.

    Note that the gate parameterizations are non-injective. For the
    decomposition to be feasible it is sufficient that one of the
    parameter values which correspond to the provided gate satisfies
    the constraints. This function will find and use the appropriate
    value whenever it exists.

    The constraints above imply that certain FSimGates are not suitable
    for use in this decomposition regardless of the target CZPowGate. We
    reject such gates based on how close |sin(θ)| is to |sin(φ/2)|, see
    atol argument below.

    This implementation accounts for the global phase.

    Args:
        cphase_gate: The CZPowGate to synthesize.
        fsim_gate: The only two qubit gate that is permitted to appear in the
            output.
        qubits: The qubits to apply the resulting operations to. If not set,
            defaults `cirq.LineQubit.range(2)`.
        atol: Tolerance used to determine whether fsim_gate is valid. The gate
            is invalid if the squares of the sines of the theta angle and half
            the phi angle are too close.

    Returns:
        Operations equivalent to cphase_gate and consisting solely of two copies
        of fsim_gate and a few single-qubit rotations.

    Raises:
        ValueError: Under any of the following circumstances:
            * cphase_gate or fsim_gate is parametrized,
            * cphase_gate and fsim_gate do not satisfy the conditions above,
            * fsim_gate has invalid angles (see atol argument above),
            * incorrect number of qubits are provided.
    