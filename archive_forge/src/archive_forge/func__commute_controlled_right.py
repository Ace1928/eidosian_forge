from typing import Sequence, Callable
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.wires import Wires
from .optimization_utils import find_next_gate
def _commute_controlled_right(op_list):
    """Push commuting single qubit gates to the right of controlled gates.

    Args:
        op_list (list[Operation]): The initial list of operations.

    Returns:
        list[Operation]: The modified list of operations with all single-qubit
        gates as far right as possible.
    """
    current_location = len(op_list) - 1
    while current_location >= 0:
        current_gate = op_list[current_location]
        if current_gate.basis is None or len(current_gate.wires) != 1:
            current_location -= 1
            continue
        next_gate_idx = find_next_gate(current_gate.wires, op_list[current_location + 1:])
        new_location = current_location
        while next_gate_idx is not None:
            next_gate = op_list[new_location + next_gate_idx + 1]
            if next_gate.basis is None:
                break
            if len(next_gate.control_wires) == 0:
                break
            shared_controls = Wires.shared_wires([Wires(current_gate.wires), next_gate.control_wires])
            if len(shared_controls) > 0:
                if current_gate.basis == 'Z':
                    new_location += next_gate_idx + 1
                else:
                    break
            elif current_gate.basis == next_gate.basis:
                new_location += next_gate_idx + 1
            else:
                break
            next_gate_idx = find_next_gate(current_gate.wires, op_list[new_location + 1:])
        op_list.insert(new_location + 1, current_gate)
        op_list.pop(current_location)
        current_location -= 1
    return op_list