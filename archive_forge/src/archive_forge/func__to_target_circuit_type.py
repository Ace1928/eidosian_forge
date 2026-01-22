from collections import defaultdict
import bisect
import dataclasses
from typing import (
from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE
def _to_target_circuit_type(circuit: circuits.AbstractCircuit, target_circuit: CIRCUIT_TYPE) -> CIRCUIT_TYPE:
    return cast(CIRCUIT_TYPE, circuit.unfreeze(copy=False) if isinstance(target_circuit, circuits.Circuit) else circuit.freeze())