from typing import (
import re
import warnings
from dataclasses import dataclass
import cirq
from cirq_google import ops
from cirq_google import transformers
from cirq_google.api import v2
from cirq_google.devices import known_devices
from cirq_google.experimental import ops as experimental_ops
def _deserialize_gateset_and_gate_durations(proto: v2.device_pb2.DeviceSpecification) -> Tuple[cirq.Gateset, Mapping[cirq.GateFamily, cirq.Duration]]:
    """Deserializes gateset and gate duration from DeviceSpecification."""
    gates_list: List[GateOrFamily] = []
    gate_durations: Dict[cirq.GateFamily, cirq.Duration] = {}
    for gate_spec in proto.valid_gates:
        gate_name = gate_spec.WhichOneof('gate')
        gate_rep = next((gr for gr in _GATES if gr.gate_spec_name == gate_name), None)
        if gate_rep is None:
            warnings.warn(f"The DeviceSpecification contains the gate '{gate_name}' which is not recognized by Cirq and will be ignored. This may be due to an out-of-date Cirq version.", UserWarning)
            continue
        gates_list.extend(gate_rep.deserialized_forms)
        for g in gate_rep.deserialized_forms:
            if not isinstance(g, cirq.GateFamily):
                g = cirq.GateFamily(g)
            gate_durations[g] = cirq.Duration(picos=gate_spec.gate_duration_picos)
    return (cirq.Gateset(*gates_list), gate_durations)