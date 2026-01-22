from __future__ import annotations
import collections.abc
import copy
from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast
import numpy as np
import pandas as pd
from xarray.core import formatting, nputils, utils
from xarray.core.indexing import (
from xarray.core.utils import (
def _check_dim_compat(variables: Mapping[Any, Variable], all_dims: str='equal'):
    """Check that all multi-index variable candidates are 1-dimensional and
    either share the same (single) dimension or each have a different dimension.

    """
    if any([var.ndim != 1 for var in variables.values()]):
        raise ValueError('PandasMultiIndex only accepts 1-dimensional variables')
    dims = {var.dims for var in variables.values()}
    if all_dims == 'equal' and len(dims) > 1:
        raise ValueError('unmatched dimensions for multi-index variables ' + ', '.join([f'{k!r} {v.dims}' for k, v in variables.items()]))
    if all_dims == 'different' and len(dims) < len(variables):
        raise ValueError('conflicting dimensions for multi-index product variables ' + ', '.join([f'{k!r} {v.dims}' for k, v in variables.items()]))