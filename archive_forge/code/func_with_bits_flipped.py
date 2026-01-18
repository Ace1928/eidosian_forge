from typing import (
import numpy as np
from cirq import _compat, protocols, value
from cirq.ops import raw_types
def with_bits_flipped(self, *bit_positions: int) -> 'MeasurementGate':
    """Toggles whether or not the measurement inverts various outputs.

        This only affects the invert_mask, which is applied after confusion
        matrices if any are defined.
        """
    old_mask = self.invert_mask or ()
    n = max(len(old_mask) - 1, *bit_positions) + 1
    new_mask = [k < len(old_mask) and old_mask[k] for k in range(n)]
    for b in bit_positions:
        new_mask[b] = not new_mask[b]
    return MeasurementGate(self.num_qubits(), key=self.key, invert_mask=tuple(new_mask), qid_shape=self._qid_shape, confusion_map=self.confusion_map)