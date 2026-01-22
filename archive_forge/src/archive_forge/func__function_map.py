from __future__ import annotations
import functools
import warnings
import numpy as np
import pandas as pd
from dask.dataframe._compat import check_to_pydatetime_deprecation
from dask.utils import derived_from
def _function_map(self, attr, *args, **kwargs):
    if 'meta' in kwargs:
        meta = kwargs.pop('meta')
    else:
        meta = self._delegate_method(self._series._meta_nonempty, self._accessor_name, attr, args, kwargs)
    token = f'{self._accessor_name}-{attr}'
    return self._series.map_partitions(self._delegate_method, self._accessor_name, attr, args, kwargs, catch_deprecation_warnings=True, meta=meta, token=token)