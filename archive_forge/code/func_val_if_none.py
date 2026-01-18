from typing import (
from cirq import protocols, value
from cirq.ops import global_phase_op, op_tree, raw_types
def val_if_none(var: Any, val: Any) -> Any:
    return var if var is not None else val