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
def _validate_device_specification(proto: v2.device_pb2.DeviceSpecification) -> None:
    """Raises a ValueError if the `DeviceSpecification` proto is invalid."""
    qubit_set = set()
    for q_name in proto.valid_qubits:
        if q_name in qubit_set:
            raise ValueError(f"Invalid DeviceSpecification: valid_qubits contains duplicate qubit '{q_name}'.")
        if re.match('^[0-9]+\\_[0-9]+$', q_name) is None:
            raise ValueError(f"Invalid DeviceSpecification: valid_qubits contains the qubit '{q_name}' which is not in the GridQubit form '<int>_<int>.")
        qubit_set.add(q_name)
    for target_set in proto.valid_targets:
        for target in target_set.targets:
            for target_id in target.ids:
                if target_id not in proto.valid_qubits:
                    raise ValueError(f"Invalid DeviceSpecification: valid_targets contain qubit '{target_id}' which is not in valid_qubits.")
        if target_set.target_ordering == v2.device_pb2.TargetSet.SYMMETRIC:
            for target in target_set.targets:
                if len(target.ids) > len(set(target.ids)):
                    raise ValueError(f"Invalid DeviceSpecification: the target set '{target_set.name}' is SYMMETRIC but has a target which contains repeated qubits: {target.ids}.")
        if target_set.target_ordering == v2.device_pb2.TargetSet.ASYMMETRIC:
            raise ValueError('Invalid DeviceSpecification: target_ordering cannot be ASYMMETRIC.')