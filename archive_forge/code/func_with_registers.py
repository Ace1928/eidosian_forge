from typing import Iterable, Sequence, Union
import attr
import cirq
import numpy as np
from cirq_ft import infra
from cirq_ft.deprecation import deprecated_cirq_ft_class
def with_registers(self, *new_registers: Union[int, Sequence[int]]) -> 'ArcTan':
    raise NotImplementedError()