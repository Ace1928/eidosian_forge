import array
from typing import (
def _do_pass(self) -> None:
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
    while 1:
        if x1 == 0:
            if self._color == 0 and self._refline[x1] == self._color:
                break
        elif x1 == len(self._refline):
            break
        elif self._refline[x1 - 1] != self._color and self._refline[x1] == self._color:
            break
        x1 += 1
    for x in range(self._curpos, x1):
        self._curline[x] = self._color
    self._curpos = x1
    return