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
def _are_list_values_of_length(array: pa.ListArray, length: int) -> bool:
    """Check if all the sub-lists of a `pa.ListArray` have the specified length."""
    return pc.all(pc.equal(array.value_lengths(), length)).as_py() or array.null_count == len(array)