import os
import numpy as np
import pytest
import cirq
from cirq.ops.pauli_interaction_gate import PauliInteractionGate
import cirq_rigetti
from cirq_rigetti.quil_output import QuilOutput
This test covers the factor of pi change. However, it will be skipped
    if pyquil is unavailable for import.

    References:
        https://docs.pytest.org/en/latest/skipping.html#skipping-on-a-missing-import-dependency
    