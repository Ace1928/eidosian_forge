import dataclasses
from typing import Callable, cast, Collection, Dict, Iterator, Optional, Sequence, Type, Union
import numpy as np
import sympy
import cirq
from cirq.devices import line_qubit
from cirq.ops import common_gates, parity_gates
from cirq_ionq.ionq_native_gates import GPIGate, GPI2Gate, MSGate
@dataclasses.dataclass
class SerializedProgram:
    """A container for the serialized portions of a `cirq.Circuit`.

    Attributes:
        body: A dictionary which contains the number of qubits and the serialized circuit
            minus the measurements.
        settings: A dictionary of settings which can override behavior for this circuit when
            run on IonQ hardware.
        metadata: A dictionary whose keys store information about the measurements in the circuit.
    """
    body: dict
    settings: dict
    metadata: dict
    error_mitigation: Optional[dict] = None