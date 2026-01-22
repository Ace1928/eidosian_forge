import abc
import dataclasses
from dataclasses import dataclass
from typing import Union, Tuple, Optional, Sequence, cast, Dict, Any, List, Iterator
import cirq
from cirq import _compat, study
@dataclass(frozen=True)
class QuantumExecutableGroup:
    """A collection of `QuantumExecutable`s.

    Attributes:
        executables: A tuple of `cg.QuantumExecutable`.
    """
    executables: Tuple[QuantumExecutable, ...]

    def __init__(self, executables: Sequence[QuantumExecutable]):
        """Initialize and normalize the quantum executable group.

        Args:
             executables: A sequence of `cg.QuantumExecutable` which will be frozen into a
                tuple.
        """
        if not isinstance(executables, tuple):
            executables = tuple(executables)
        object.__setattr__(self, 'executables', executables)
        object.__setattr__(self, '_hash', hash(dataclasses.astuple(self)))

    def __len__(self) -> int:
        return len(self.executables)

    def __iter__(self) -> Iterator[QuantumExecutable]:
        yield from self.executables

    def __str__(self) -> str:
        exe_str = ', '.join((str(exe) for exe in self.executables[:2]))
        if len(self.executables) > 2:
            exe_str += ', ...'
        return f'QuantumExecutableGroup(executables=[{exe_str}])'

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')

    def __hash__(self) -> int:
        return self._hash

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.dataclass_json_dict(self)