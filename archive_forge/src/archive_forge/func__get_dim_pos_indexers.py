from __future__ import annotations
import functools
import operator
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable, Final, Generic, TypeVar, cast, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes
from xarray.core.indexes import (
from xarray.core.types import T_Alignable
from xarray.core.utils import is_dict_like, is_full_slice
from xarray.core.variable import Variable, as_compatible_data, calculate_dimensions
def _get_dim_pos_indexers(self, matching_indexes: dict[MatchingIndexKey, Index]) -> dict[Hashable, Any]:
    dim_pos_indexers = {}
    for key, aligned_idx in self.aligned_indexes.items():
        obj_idx = matching_indexes.get(key)
        if obj_idx is not None:
            if self.reindex[key]:
                indexers = obj_idx.reindex_like(aligned_idx, **self.reindex_kwargs)
                dim_pos_indexers.update(indexers)
    return dim_pos_indexers