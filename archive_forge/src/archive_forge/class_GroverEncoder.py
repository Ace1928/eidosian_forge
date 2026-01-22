from typing import Optional, Sequence, Tuple
import cirq
import cirq_ft
import numpy as np
import pytest
from attr import frozen
from cirq_ft import infra
from cirq._compat import cached_property
from cirq_ft.algos.mean_estimation import CodeForRandomVariable, MeanEstimationOperator
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@frozen
class GroverEncoder(cirq_ft.SelectOracle):
    """Enc|marked_item>|0> --> |marked_item>|marked_val>"""
    n: int
    marked_item: int
    marked_val: int

    @cached_property
    def control_registers(self) -> Tuple[cirq_ft.Register, ...]:
        return ()

    @cached_property
    def selection_registers(self) -> Tuple[cirq_ft.SelectionRegister, ...]:
        return (cirq_ft.SelectionRegister('selection', self.n),)

    @cached_property
    def target_registers(self) -> Tuple[cirq_ft.Register, ...]:
        return (cirq_ft.Register('target', self.marked_val.bit_length()),)

    def decompose_from_registers(self, context, *, selection: Sequence[cirq.Qid], target: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        selection_cv = [*bit_tools.iter_bits(self.marked_item, infra.total_bits(self.selection_registers))]
        yval_bin = [*bit_tools.iter_bits(self.marked_val, infra.total_bits(self.target_registers))]
        for b, q in zip(yval_bin, target):
            if b:
                yield cirq.X(q).controlled_by(*selection, control_values=selection_cv)

    @cached_property
    def mu(self) -> float:
        return self.marked_val / 2 ** self.n

    @cached_property
    def s_square(self) -> float:
        return self.marked_val ** 2 / 2 ** self.n