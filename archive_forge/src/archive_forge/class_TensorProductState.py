import logging
import re
from dataclasses import dataclass
from typing import Any, FrozenSet, Generator, Iterable, List, Optional, cast
from pyquil.paulis import PauliTerm, sI
@dataclass(frozen=True)
class TensorProductState:
    """
    A description of a multi-qubit quantum state that is a tensor product of many _OneQStates
    states.
    """
    states: List[_OneQState]

    def __init__(self, states: Optional[Iterable[_OneQState]]=None):
        if states is None:
            states = []
        object.__setattr__(self, 'states', list(states))

    def __mul__(self, other: 'TensorProductState') -> 'TensorProductState':
        return TensorProductState(self.states + other.states)

    def __str__(self) -> str:
        return ' * '.join((str(s) for s in self.states))

    def __repr__(self) -> str:
        return f'TensorProductState[{self}]'

    def __getitem__(self, qubit: int) -> _OneQState:
        """Return the _OneQState at the given qubit."""
        for oneq_state in self.states:
            if oneq_state.qubit == qubit:
                return oneq_state
        raise IndexError()

    def __iter__(self) -> Generator[_OneQState, None, None]:
        yield from self.states

    def __len__(self) -> int:
        return len(self.states)

    def states_as_set(self) -> FrozenSet[_OneQState]:
        return frozenset(self.states)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TensorProductState):
            return False
        return self.states_as_set() == other.states_as_set()

    def __hash__(self) -> int:
        return hash(self.states_as_set())

    @classmethod
    def from_str(cls, s: str) -> 'TensorProductState':
        if s == '':
            return TensorProductState()
        return TensorProductState(list((_OneQState.from_str(x) for x in s.split('*'))))