from __future__ import annotations
from collections.abc import (
import csv as csvlib
import os
from typing import (
import numpy as np
from pandas._libs import writers as libwriters
from pandas._typing import SequenceNotStr
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import notna
from pandas.core.indexes.api import Index
from pandas.io.common import get_handle
def _save_body(self) -> None:
    nrows = len(self.data_index)
    chunks = nrows // self.chunksize + 1
    for i in range(chunks):
        start_i = i * self.chunksize
        end_i = min(start_i + self.chunksize, nrows)
        if start_i >= end_i:
            break
        self._save_chunk(start_i, end_i)