from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
@classmethod
def _get_sqrt_map(cls) -> Dict[float, Dict['SingleQubitCliffordGate', 'SingleQubitCliffordGate']]:
    """Returns a map containing two keys 0.5 and -0.5 for the sqrt mapping of Pauli gates."""
    return {0.5: {cls.X: cls.X_sqrt, cls.Y: cls.Y_sqrt, cls.Z: cls.Z_sqrt}, -0.5: {cls.X: cls.X_nsqrt, cls.Y: cls.Y_nsqrt, cls.Z: cls.Z_nsqrt}}