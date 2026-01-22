from __future__ import annotations
from typing import (
from simpy.exceptions import Interrupt
def _desc(self) -> str:
    """Return a string *Condition(evaluate, [events])*."""
    return f'{self.__class__.__name__}({self._evaluate.__name__}, {self._events})'