from __future__ import annotations
import copy
import datetime
import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.cftime_offsets import _new_to_legacy_freq
from xarray.core import dtypes, duck_array_ops, nputils, ops
from xarray.core._aggregations import (
from xarray.core.alignment import align
from xarray.core.arithmetic import DataArrayGroupbyArithmetic, DatasetGroupbyArithmetic
from xarray.core.common import ImplementsArrayReduce, ImplementsDatasetReduce
from xarray.core.concat import concat
from xarray.core.formatting import format_array_flat
from xarray.core.indexes import (
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import (
from xarray.core.utils import (
from xarray.core.variable import IndexVariable, Variable
from xarray.util.deprecation_helpers import _deprecate_positional_args
@dataclass
class BinGrouper(Grouper):
    """Grouper object for binning numeric data."""
    bins: Any
    cut_kwargs: Mapping = field(default_factory=dict)
    binned: Any = None
    name: Any = None

    def __post_init__(self) -> None:
        if duck_array_ops.isnull(self.bins).all():
            raise ValueError('All bin edges are NaN.')

    def factorize(self, group) -> EncodedGroups:
        from xarray.core.dataarray import DataArray
        data = group.data
        binned, self.bins = pd.cut(data, self.bins, **self.cut_kwargs, retbins=True)
        binned_codes = binned.codes
        if (binned_codes == -1).all():
            raise ValueError(f'None of the data falls within bins with edges {self.bins!r}')
        new_dim_name = f'{group.name}_bins'
        full_index = binned.categories
        uniques = np.sort(pd.unique(binned_codes))
        unique_values = full_index[uniques[uniques != -1]]
        codes = DataArray(binned_codes, getattr(group, 'coords', None), name=new_dim_name)
        unique_coord = IndexVariable(new_dim_name, pd.Index(unique_values), group.attrs)
        return EncodedGroups(codes=codes, full_index=full_index, unique_coord=unique_coord)