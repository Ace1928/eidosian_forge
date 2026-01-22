from __future__ import annotations
import collections
from typing import Callable, Iterable, TYPE_CHECKING
from coverage.debug import auto_repr
from coverage.exceptions import ConfigError
from coverage.misc import nice_pair
from coverage.types import TArc, TLineNo
def arcs_missing(self) -> list[TArc]:
    """Returns a sorted list of the un-executed arcs in the code."""
    possible = self.arc_possibilities()
    executed = self.arcs_executed()
    missing = (p for p in possible if p not in executed and p[0] not in self.no_branch and (p[1] not in self.excluded))
    return sorted(missing)