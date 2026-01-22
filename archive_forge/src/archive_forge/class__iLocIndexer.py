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
class _iLocIndexer(_IndexerBase):

    @property
    def _meta_indexer(self):
        return self.obj._meta.iloc

    def __getitem__(self, key):
        msg = "'DataFrame.iloc' only supports selecting columns. It must be used like 'df.iloc[:, column_indexer]'."
        if not isinstance(key, tuple):
            raise NotImplementedError(msg)
        if len(key) > 2:
            raise ValueError('Too many indexers')
        iindexer, cindexer = key
        if iindexer != slice(None):
            raise NotImplementedError(msg)
        if not self.obj.columns.is_unique:
            return self._iloc(iindexer, cindexer)
        else:
            col_names = self.obj.columns[cindexer]
            return self.obj.__getitem__(col_names)

    def _iloc(self, iindexer, cindexer):
        assert iindexer == slice(None)
        meta = self._make_meta(iindexer, cindexer)
        return self.obj.map_partitions(methods.iloc, cindexer, meta=meta)