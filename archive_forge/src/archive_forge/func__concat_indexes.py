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
@staticmethod
def _concat_indexes(indexes, dim, positions=None) -> pd.Index:
    new_pd_index: pd.Index
    if not indexes:
        new_pd_index = pd.Index([])
    else:
        if not all((idx.dim == dim for idx in indexes)):
            dims = ','.join({f'{idx.dim!r}' for idx in indexes})
            raise ValueError(f'Cannot concatenate along dimension {dim!r} indexes with dimensions: {dims}')
        pd_indexes = [idx.index for idx in indexes]
        new_pd_index = pd_indexes[0].append(pd_indexes[1:])
        if positions is not None:
            indices = nputils.inverse_permutation(np.concatenate(positions))
            new_pd_index = new_pd_index.take(indices)
    return new_pd_index