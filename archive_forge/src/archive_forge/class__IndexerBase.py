from __future__ import annotations
import bisect
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from dask.array.core import Array
from dask.base import tokenize
from dask.dataframe import methods
from dask.dataframe._compat import IndexingError
from dask.dataframe.core import Series, new_dd_object
from dask.dataframe.utils import is_index_like, is_series_like, meta_nonempty
from dask.highlevelgraph import HighLevelGraph
from dask.utils import is_arraylike
class _IndexerBase:

    def __init__(self, obj):
        self.obj = obj

    @property
    def _name(self):
        return self.obj._name

    @property
    def _meta_indexer(self):
        raise NotImplementedError

    def _make_meta(self, iindexer, cindexer):
        """
        get metadata
        """
        if cindexer is None:
            return self.obj
        else:
            return self._meta_indexer[:, cindexer]

    def __dask_tokenize__(self):
        return (type(self).__name__, tokenize(self.obj))