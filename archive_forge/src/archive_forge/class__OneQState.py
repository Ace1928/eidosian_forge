import logging
import re
from dataclasses import dataclass
from typing import Any, FrozenSet, Generator, Iterable, List, Optional, cast
from pyquil.paulis import PauliTerm, sI
@dataclass(frozen=True)
class _OneQState:
    """
    A description of a named one-qubit quantum state.

    This can be used to generate pre-rotations for quantum process tomography. For example,
    X0_14 will generate the +1 eigenstate of the X operator on qubit 14. X1_14 will generate the
    -1 eigenstate. SIC0_14 will generate the 0th SIC-basis state on qubit 14.
    """
    label: str
    index: int
    qubit: int

    def __str__(self) -> str:
        return f'{self.label}{self.index}_{self.qubit}'

    @classmethod
    def from_str(cls, s: str) -> '_OneQState':
        ma = re.match('\\s*(\\w+)(\\d+)_(\\d+)\\s*', s)
        if ma is None:
            raise ValueError(f"Couldn't parse '{s}'")
        return _OneQState(label=ma.group(1), index=int(ma.group(2)), qubit=int(ma.group(3)))