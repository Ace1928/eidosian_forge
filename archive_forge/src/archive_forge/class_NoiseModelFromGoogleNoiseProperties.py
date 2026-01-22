import dataclasses
from typing import Any, Dict, List, Sequence, Set, Type, TypeVar, Union
import numpy as np
import cirq, cirq_google
from cirq import _compat, devices
from cirq.devices import noise_utils
from cirq.transformers.heuristic_decompositions import gate_tabulation_math_utils
class NoiseModelFromGoogleNoiseProperties(devices.NoiseModelFromNoiseProperties):
    """A noise model defined from noise properties of a Google device."""

    def is_virtual(self, op: cirq.Operation) -> bool:
        return isinstance(op.gate, cirq.ZPowGate) and cirq_google.PhysicalZTag not in op.tags