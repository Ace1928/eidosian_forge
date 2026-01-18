import copy
import os
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from . import config
from .utils.logging import get_logger
def to_batches(self, *args, **kwargs):
    """
        Convert Table to list of (contiguous) `RecordBatch` objects.

        Args:
            max_chunksize (`int`, defaults to `None`):
                Maximum size for `RecordBatch` chunks. Individual chunks may be
                smaller depending on the chunk layout of individual columns.

        Returns:
            `List[pyarrow.RecordBatch]`
        """
    return self.table.to_batches(*args, **kwargs)