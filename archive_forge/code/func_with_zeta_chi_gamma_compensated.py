import abc
import collections
import dataclasses
import functools
import math
import re
from typing import (
import numpy as np
import pandas as pd
import cirq
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.engine import (
from cirq_google.ops import FSimGateFamily, SycamoreGate
def with_zeta_chi_gamma_compensated(self, qubits: Tuple[cirq.Qid, cirq.Qid], parameters: PhasedFSimCharacterization, *, engine_gate: Optional[cirq.Gate]=None) -> Tuple[Tuple[cirq.Operation, ...], ...]:
    """Creates a composite operation that compensates for zeta, chi and gamma angles of the
        characterization.

        Args:
            qubits: Qubits that the gate should act on.
            parameters: The results of characterization of the engine gate.
            engine_gate: 2-qubit gate that represents the engine gate. When None, the internal
                engine_gate of this instance is used. This argument is useful for testing
                purposes.

        Returns:
            Tuple of tuple of operations that describe the compensated gate. The first index
            iterates over moments of the composed operation.

        Raises:
            ValueError: If the engine gate is not a 2-qubit gate.
        """
    assert parameters.zeta is not None, 'Zeta value must not be None'
    zeta = parameters.zeta
    assert parameters.gamma is not None, 'Gamma value must not be None'
    gamma = parameters.gamma
    assert parameters.chi is not None, 'Chi value must not be None'
    chi = parameters.chi + 2 * np.pi * self.phase_exponent
    if engine_gate is None:
        engine_gate = self.engine_gate
    elif cirq.num_qubits(engine_gate) != 2:
        raise ValueError('Engine gate must be a two-qubit gate')
    a, b = qubits
    alpha = 0.5 * (zeta + chi)
    beta = 0.5 * (zeta - chi)
    return ((cirq.rz(0.5 * gamma - alpha).on(a), cirq.rz(0.5 * gamma + alpha).on(b)), (engine_gate.on(a, b),), (cirq.rz(0.5 * gamma - beta).on(a), cirq.rz(0.5 * gamma + beta).on(b)))