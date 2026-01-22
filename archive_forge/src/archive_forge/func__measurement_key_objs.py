import math
from typing import (
import numpy as np
import sympy
from cirq import circuits, ops, protocols, value, study
from cirq._compat import cached_property, proper_repr
@cached_property
def _measurement_key_objs(self) -> FrozenSet['cirq.MeasurementKey']:
    circuit_keys = protocols.measurement_key_objs(self.circuit)
    if circuit_keys and self.use_repetition_ids:
        self._ensure_deterministic_loop_count()
        if self.repetition_ids is not None:
            circuit_keys = frozenset((key.with_key_path_prefix(repetition_id) for repetition_id in self.repetition_ids for key in circuit_keys))
    circuit_keys = frozenset((key.with_key_path_prefix(*self.parent_path) for key in circuit_keys))
    return frozenset((protocols.with_measurement_key_mapping(key, self.measurement_key_map) for key in circuit_keys))