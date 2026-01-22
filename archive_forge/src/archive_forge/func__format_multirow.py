from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
def _format_multirow(self, row: list[str], i: int) -> list[str]:
    """
        Check following rows, whether row should be a multirow

        e.g.:     becomes:
        a & 0 &   \\multirow{2}{*}{a} & 0 &
          & 1 &     & 1 &
        b & 0 &   \\cline{1-2}
                  b & 0 &
        """
    for j in range(self.index_levels):
        if row[j].strip():
            nrow = 1
            for r in self.strrows[i + 1:]:
                if not r[j].strip():
                    nrow += 1
                else:
                    break
            if nrow > 1:
                row[j] = f'\\multirow{{{nrow:d}}}{{*}}{{{row[j].strip()}}}'
                self.clinebuf.append([i + nrow - 1, j + 1])
    return row