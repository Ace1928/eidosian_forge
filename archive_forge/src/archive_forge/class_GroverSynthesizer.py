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
class GroverSynthesizer(cirq_ft.PrepareOracle):
    """Prepare a uniform superposition over the first $2^n$ elements."""
    n: int

    @cached_property
    def selection_registers(self) -> Tuple[cirq_ft.SelectionRegister, ...]:
        return (cirq_ft.SelectionRegister('selection', self.n),)

    def decompose_from_registers(self, *, context, selection: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        yield cirq.H.on_each(*selection)

    def __pow__(self, power):
        if power in [+1, -1]:
            return self
        return NotImplemented