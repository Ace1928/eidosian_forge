import dataclasses
from typing import Callable, cast, Collection, Dict, Iterator, Optional, Sequence, Type, Union
import numpy as np
import sympy
import cirq
from cirq.devices import line_qubit
from cirq.ops import common_gates, parity_gates
from cirq_ionq.ionq_native_gates import GPIGate, GPI2Gate, MSGate
def _serialize_z_pow_gate(self, gate: cirq.ZPowGate, targets: Sequence[int]) -> dict:
    if self._near_mod_n(gate.exponent, 1, 2):
        return {'gate': 'z', 'targets': targets}
    elif self._near_mod_n(gate.exponent, 0.5, 2):
        return {'gate': 's', 'targets': targets}
    elif self._near_mod_n(gate.exponent, -0.5, 2):
        return {'gate': 'si', 'targets': targets}
    elif self._near_mod_n(gate.exponent, 0.25, 2):
        return {'gate': 't', 'targets': targets}
    elif self._near_mod_n(gate.exponent, -0.25, 2):
        return {'gate': 'ti', 'targets': targets}
    return {'gate': 'rz', 'targets': targets, 'rotation': gate.exponent * np.pi}