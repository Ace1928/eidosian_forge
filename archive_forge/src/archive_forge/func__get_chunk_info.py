from __future__ import annotations
import copy
import linecache
from io import StringIO
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.io.pwmat.inputs import ACstrExtractor, AtomConfig, LineLocator
def _get_chunk_info(self) -> tuple[list[int], list[int]]:
    """Split MOVEMENT into many chunks, so that program process it chunk by chunk.

        Returns:
            tuple[list[int], list[int]]:
                chunk_sizes (list[int]): The number of lines occupied by structural
                    information in each step.
                chunk_starts (list[int]): The starting line number for structural
                    information in each step.
        """
    chunk_sizes: list[int] = []
    row_idxs: list[int] = LineLocator.locate_all_lines(self.filename, self.split_mark)
    chunk_sizes.append(row_idxs[0])
    for ii in range(1, len(row_idxs)):
        chunk_sizes.append(row_idxs[ii] - row_idxs[ii - 1])
    chunk_sizes_bak: list[int] = copy.deepcopy(chunk_sizes)
    chunk_sizes_bak.insert(0, 0)
    chunk_starts: list[int] = np.cumsum(chunk_sizes_bak).tolist()
    chunk_starts.pop(-1)
    return (chunk_sizes, chunk_starts)