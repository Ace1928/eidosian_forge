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
def _serialize_gateset_and_gate_durations(out: v2.device_pb2.DeviceSpecification, gateset: cirq.Gateset, gate_durations: Mapping[cirq.GateFamily, cirq.Duration]) -> v2.device_pb2.DeviceSpecification:
    """Serializes the given gateset and gate durations to DeviceSpecification."""
    gate_specs: Dict[str, v2.device_pb2.GateSpecification] = {}
    for gate_family in gateset.gates:
        gate_spec = v2.device_pb2.GateSpecification()
        gate_rep = next((gr for gr in _GATES for gf in gr.serializable_forms if gf == gate_family), None)
        if gate_rep is None:
            raise ValueError(f'Unrecognized gate: {gate_family}.')
        gate_name = gate_rep.gate_spec_name
        getattr(gate_spec, gate_name).SetInParent()
        gate_durations_picos = {int(gate_durations[gf].total_picos()) for gf in gate_rep.serializable_forms if gf in gate_durations}
        if len(gate_durations_picos) > 1:
            raise ValueError(f'Multiple gate families in the following list exist in the gate duration dict, and they are expected to have the same duration value: {gate_rep.serializable_forms}')
        elif len(gate_durations_picos) == 1:
            gate_spec.gate_duration_picos = gate_durations_picos.pop()
        gate_specs[gate_name] = gate_spec
    out.valid_gates.extend((v for _, v in sorted(gate_specs.items())))
    return out