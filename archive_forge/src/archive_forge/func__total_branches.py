from __future__ import annotations
import collections
from typing import Callable, Iterable, TYPE_CHECKING
from coverage.debug import auto_repr
from coverage.exceptions import ConfigError
from coverage.misc import nice_pair
from coverage.types import TArc, TLineNo
def _total_branches(self) -> int:
    """How many total branches are there?"""
    return sum((count for count in self.exit_counts.values() if count > 1))