import logging
import re
from dataclasses import dataclass
from typing import Any, FrozenSet, Generator, Iterable, List, Optional, cast
from pyquil.paulis import PauliTerm, sI
def zeros_state(qubits: Iterable[int]) -> TensorProductState:
    return TensorProductState([_OneQState(label='Z', index=0, qubit=q) for q in qubits])