from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import cirq
from cirq import ops, linalg, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker
@value.value_equality
class QuirkInputRotationOperation(ops.Operation):
    """Operates on target qubits in a way that varies based on an input qureg."""

    def __init__(self, identifier: str, register: Iterable['cirq.Qid'], base_operation: 'cirq.Operation', exponent_sign: int):
        if exponent_sign not in [-1, +1]:
            raise ValueError('exponent_sign not in [-1, +1]')
        self.identifier = identifier
        self.register = tuple(register)
        self.base_operation = base_operation
        self.exponent_sign = exponent_sign

    def _value_equality_values_(self) -> Any:
        return (self.identifier, self.register, self.base_operation, self.exponent_sign)

    @property
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        return tuple(self.base_operation.qubits) + self.register

    def with_qubits(self, *new_qubits):
        k = len(self.base_operation.qubits)
        new_op_qubits = new_qubits[:k]
        new_register = new_qubits[k:]
        return QuirkInputRotationOperation(self.identifier, new_register, self.base_operation.with_qubits(*new_op_qubits), self.exponent_sign)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'):
        sub_result = cirq.circuit_diagram_info(self.base_operation)
        sign_char = '-' if self.exponent_sign == -1 else ''
        symbols = list(sub_result.wire_symbols)
        symbols.extend((f'A{i}' for i in range(len(self.register))))
        return cirq.CircuitDiagramInfo(tuple(symbols), exponent=f'({sign_char}A/2^{len(self.register)})', exponent_qubit_index=sub_result.exponent_qubit_index or 0, auto_exponent_parens=False)

    def _has_unitary_(self) -> bool:
        return True

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs'):
        transposed_args = args.with_axes_transposed_to_start()
        target_axes = transposed_args.axes[:len(self.base_operation.qubits)]
        control_axes = transposed_args.axes[len(self.base_operation.qubits):]
        control_max = np.prod([q.dimension for q in self.register], dtype=np.int64).item()
        for i in range(control_max):
            assert isinstance(self.base_operation, cirq.GateOperation)
            operation = self.base_operation ** (self.exponent_sign * i / control_max)
            control_index = linalg.slice_for_qubits_equal_to(control_axes, big_endian_qureg_value=i)
            sub_args = cirq.ApplyUnitaryArgs(transposed_args.target_tensor[control_index], transposed_args.available_buffer[control_index], target_axes)
            sub_result = cirq.apply_unitary(operation, sub_args)
            if sub_result is not sub_args.target_tensor:
                sub_args.target_tensor[...] = sub_result
        return args.target_tensor

    def __repr__(self) -> str:
        return f'cirq.interop.quirk.QuirkInputRotationOperation(identifier={self.identifier!r}, register={self.register!r}, base_operation={self.base_operation!r}, exponent_sign={self.exponent_sign!r})'