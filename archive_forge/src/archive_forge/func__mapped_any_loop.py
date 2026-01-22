import math
from typing import (
import numpy as np
import sympy
from cirq import circuits, ops, protocols, value, study
from cirq._compat import cached_property, proper_repr
@cached_property
def _mapped_any_loop(self) -> 'cirq.Circuit':
    circuit = self.circuit.unfreeze()
    if self.qubit_map:
        circuit = circuit.transform_qubits(lambda q: self.qubit_map.get(q, q))
    if isinstance(self.repetitions, INT_CLASSES) and self.repetitions < 0:
        circuit = circuit ** (-1)
    if self.measurement_key_map:
        circuit = protocols.with_measurement_key_mapping(circuit, self.measurement_key_map)
    if self.param_resolver:
        circuit = protocols.resolve_parameters(circuit, self.param_resolver, recursive=False)
    return circuit.unfreeze(copy=False)