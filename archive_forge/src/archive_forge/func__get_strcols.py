from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
def _get_strcols(self) -> list[list[str]]:
    """String representation of the columns."""
    if self.fmt.frame.empty:
        strcols = [[self._empty_info_line]]
    else:
        strcols = self.fmt.get_strcols()
    if self.fmt.index and isinstance(self.frame.index, ABCMultiIndex):
        out = self.frame.index.format(adjoin=False, sparsify=self.fmt.sparsify, names=self.fmt.has_index_names, na_rep=self.fmt.na_rep)

        def pad_empties(x):
            for pad in reversed(x):
                if pad:
                    return [x[0]] + [i if i else ' ' * len(pad) for i in x[1:]]
        gen = (pad_empties(i) for i in out)
        clevels = self.frame.columns.nlevels
        out = [[' ' * len(i[-1])] * clevels + i for i in gen]
        cnames = self.frame.columns.names
        if any(cnames):
            new_names = [i if i else '{}' for i in cnames]
            out[self.frame.index.nlevels - 1][:clevels] = new_names
        strcols = out + strcols[1:]
    return strcols