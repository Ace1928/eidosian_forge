import array
from typing import (
def _do_horizontal(self, n1: int, n2: int) -> None:
    if self._curpos < 0:
        self._curpos = 0
    x = self._curpos
    for _ in range(n1):
        if len(self._curline) <= x:
            break
        self._curline[x] = self._color
        x += 1
    for _ in range(n2):
        if len(self._curline) <= x:
            break
        self._curline[x] = 1 - self._color
        x += 1
    self._curpos = x
    return