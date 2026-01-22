from __future__ import annotations
from typing import Literal
import numpy as np
from pandas.compat import pa_version_under10p1
def _str_slice_replace(self, start: int | None=None, stop: int | None=None, repl: str | None=None):
    if repl is None:
        repl = ''
    if start is None:
        start = 0
    if stop is None:
        stop = np.iinfo(np.int64).max
    return type(self)(pc.utf8_replace_slice(self._pa_array, start, stop, repl))