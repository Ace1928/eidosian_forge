from typing import Any, Callable, Hashable, Iterable, Optional, Union, overload
import attr
import cachetools
import cirq
from cirq_ft.infra.decompose_protocol import _decompose_once_considering_known_decomposition
from typing_extensions import Literal, Protocol
from cirq_ft.deprecation import deprecated_cirq_ft_class, deprecated_cirq_ft_function
@deprecated_cirq_ft_class()
@attr.frozen
class TComplexity:
    """Dataclass storing counts of logical T-gates, Clifford gates and single qubit rotations."""
    t: int = 0
    clifford: int = 0
    rotations: int = 0

    def __add__(self, other: 'TComplexity') -> 'TComplexity':
        return TComplexity(self.t + other.t, self.clifford + other.clifford, self.rotations + other.rotations)

    def __mul__(self, other: int) -> 'TComplexity':
        return TComplexity(self.t * other, self.clifford * other, self.rotations * other)

    def __rmul__(self, other: int) -> 'TComplexity':
        return self.__mul__(other)

    def __str__(self) -> str:
        return f'T-count:   {self.t:g}\nRotations: {self.rotations:g}\nCliffords: {self.clifford:g}\n'