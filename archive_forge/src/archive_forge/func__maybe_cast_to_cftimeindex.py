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
def _maybe_cast_to_cftimeindex(index: pd.Index) -> pd.Index:
    from xarray.coding.cftimeindex import CFTimeIndex
    if len(index) > 0 and index.dtype == 'O' and (not isinstance(index, CFTimeIndex)):
        try:
            return CFTimeIndex(index)
        except (ImportError, TypeError):
            return index
    else:
        return index