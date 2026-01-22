import logging
import re
from dataclasses import dataclass
from typing import Any, FrozenSet, Generator, Iterable, List, Optional, cast
from pyquil.paulis import PauliTerm, sI
def SIC3(q: int) -> TensorProductState:
    return TensorProductState([_OneQState(label='SIC', index=3, qubit=q)])