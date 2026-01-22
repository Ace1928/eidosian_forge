import array
from typing import (
def _do_vertical(self, dx: int) -> None:
    x1 = self._curpos + 1
    while 1:
        if x1 == 0:
            if self._color == 1 and self._refline[x1] != self._color:
                break
        elif x1 == len(self._refline):
            break
        elif self._refline[x1 - 1] == self._color and self._refline[x1] != self._color:
            break
        x1 += 1
    x1 += dx
    x0 = max(0, self._curpos)
    x1 = max(0, min(self.width, x1))
    if x1 < x0:
        for x in range(x1, x0):
            self._curline[x] = self._color
    elif x0 < x1:
        for x in range(x0, x1):
            self._curline[x] = self._color
    self._curpos = x1
    self._color = 1 - self._color
    return