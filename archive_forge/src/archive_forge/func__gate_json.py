from typing import (
from cirq import protocols, value
from cirq.ops import global_phase_op, op_tree, raw_types
def _gate_json(self) -> Union[raw_types.Gate, str]:
    return self.gate if not isinstance(self.gate, type) else protocols.json_cirq_type(self.gate)