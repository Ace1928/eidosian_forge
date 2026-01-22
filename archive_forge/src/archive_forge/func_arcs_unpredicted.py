from __future__ import annotations
import collections
from typing import Callable, Iterable, TYPE_CHECKING
from coverage.debug import auto_repr
from coverage.exceptions import ConfigError
from coverage.misc import nice_pair
from coverage.types import TArc, TLineNo
def arcs_unpredicted(self) -> list[TArc]:
    """Returns a sorted list of the executed arcs missing from the code."""
    possible = self.arc_possibilities()
    executed = self.arcs_executed()
    unpredicted = (e for e in executed if e not in possible and e[0] != e[1] and (e[0] > 0 or e[1] > 0))
    return sorted(unpredicted)