import math
from typing import Optional, Tuple
import cirq
import cirq_ft
import numpy as np
import pytest
from attr import frozen
from cirq._compat import cached_property
from cirq_ft.algos.mean_estimation.complex_phase_oracle import ComplexPhaseOracle
from cirq_ft.infra import bit_tools
from cirq_ft.infra import testing as cq_testing
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@frozen
class ExampleSelect(cirq_ft.SelectOracle):
    bitsize: int
    control_val: Optional[int] = None

    @cached_property
    def control_registers(self) -> Tuple[cirq_ft.Register, ...]:
        return () if self.control_val is None else (cirq_ft.Register('control', 1),)

    @cached_property
    def selection_registers(self) -> Tuple[cirq_ft.SelectionRegister, ...]:
        return (cirq_ft.SelectionRegister('selection', self.bitsize),)

    @cached_property
    def target_registers(self) -> Tuple[cirq_ft.Register, ...]:
        return (cirq_ft.Register('target', self.bitsize),)

    def decompose_from_registers(self, context, selection, target):
        yield [cirq.CNOT(s, t) for s, t in zip(selection, target)]