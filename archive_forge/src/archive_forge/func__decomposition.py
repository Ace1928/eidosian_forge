from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import attr
import cirq
from cirq._compat import cached_property
from numpy.typing import NDArray
from cirq_ft import infra
from cirq_ft.algos import and_gate
from cirq_ft.deprecation import deprecated_cirq_ft_class
def _decomposition() -> Iterator[cirq.Operation]:
    yield and_gate.And((0, 1), adjoint=self.adjoint).on(*a, *b, *less_than)
    yield cirq.CNOT(*less_than, *greater_than)
    yield cirq.CNOT(*b, *greater_than)
    yield cirq.CNOT(*a, *b)
    yield cirq.CNOT(*a, *greater_than)
    yield cirq.X(*b)