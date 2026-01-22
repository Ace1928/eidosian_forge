from __future__ import annotations
import copy
import math
import sys
import warnings
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import (
import numpy as np
from xarray.core import dtypes, formatting, formatting_html
from xarray.core.indexing import (
from xarray.namedarray._aggregations import NamedArrayAggregations
from xarray.namedarray._typing import (
from xarray.namedarray.parallelcompat import guess_chunkmanager
from xarray.namedarray.pycompat import to_numpy
from xarray.namedarray.utils import (
def _raise_if_any_duplicate_dimensions(dims: _Dims, err_context: str='This function') -> None:
    if len(set(dims)) < len(dims):
        repeated_dims = {d for d in dims if dims.count(d) > 1}
        raise ValueError(f"{err_context} cannot handle duplicate dimensions, but dimensions {repeated_dims} appear more than once on this object's dims: {dims}")