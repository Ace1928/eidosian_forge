import abc
from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING, List, Set, Type
from cirq import _compat, ops, devices
from cirq.devices import noise_utils
def _validate_symmetric_errors(self, field_name: str) -> None:
    gate_error_dict = getattr(self, field_name)
    for op_id in gate_error_dict:
        if len(op_id.qubits) != 2:
            if len(op_id.qubits) > 2:
                raise ValueError(f'Found gate {op_id.gate_type} with {len(op_id.qubits)} qubits. Symmetric errors can only apply to 2-qubit gates.')
        elif op_id.gate_type in self.symmetric_two_qubit_gates():
            op_id_swapped = noise_utils.OpIdentifier(op_id.gate_type, *op_id.qubits[::-1])
            if op_id_swapped not in gate_error_dict:
                raise ValueError(f'Operation {op_id} of field {field_name} has errors but its symmetric id {op_id_swapped} does not.')
        elif op_id.gate_type not in self.asymmetric_two_qubit_gates():
            raise ValueError(f'Found gate {op_id.gate_type} which does not appear in the symmetric or asymmetric gate sets.')