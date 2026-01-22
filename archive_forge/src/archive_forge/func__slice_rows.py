import math
from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar
import numpy as np
import pandas as pd
import pyarrow as pa
from triad.utils.convert import to_size
from .iter import slice_iterable
def _slice_rows(self, batch_rows: int, initial_rows: int, slice_rows: int) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
    start = 0
    if initial_rows >= batch_rows:
        return ([(0, batch_rows)], (0, 0))
    slices = [(0, initial_rows)]
    start = initial_rows
    while True:
        if batch_rows - start < slice_rows:
            return (slices, (start, batch_rows - start))
        slices.append((start, slice_rows))
        start += slice_rows