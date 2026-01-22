from __future__ import annotations
import copy
import pickle
import threading
import warnings
from collections import OrderedDict, defaultdict
from contextlib import ExitStack
import numpy as np
import pandas as pd
import tlz as toolz
from packaging.version import parse as parse_version
from dask.core import flatten
from dask.dataframe._compat import PANDAS_GE_201
from dask.base import tokenize
from dask.dataframe.io.parquet.utils import (
from dask.dataframe.io.utils import _is_local_fs, _meta_from_dtypes, _open_input_files
from dask.dataframe.utils import UNKNOWN_CATEGORIES
from dask.delayed import Delayed
from dask.utils import natural_sort_key
def _paths_to_cats(paths, file_scheme):
    """
    Extract categorical fields and labels from hive- or drill-style paths.
    FixMe: This has been pasted from https://github.com/dask/fastparquet/pull/471
    Use fastparquet.api.paths_to_cats from fastparquet>0.3.2 instead.

    Parameters
    ----------
    paths (Iterable[str]): file paths relative to root
    file_scheme (str):

    Returns
    -------
    cats (OrderedDict[str, List[Any]]): a dict of field names and their values
    """
    if file_scheme in ['simple', 'flat', 'other']:
        cats = {}
        return cats
    cats = OrderedDict()
    raw_cats = OrderedDict()
    s = ex_from_sep('/')
    paths = toolz.unique(paths)
    if file_scheme == 'hive':
        partitions = toolz.unique(((k, v) for path in paths for k, v in s.findall(path)))
        for key, val in partitions:
            cats.setdefault(key, set()).add(val_to_num(val))
            raw_cats.setdefault(key, set()).add(val)
    else:
        i_val = toolz.unique(((i, val) for path in paths for i, val in enumerate(path.split('/')[:-1])))
        for i, val in i_val:
            key = 'dir%i' % i
            cats.setdefault(key, set()).add(val_to_num(val))
            raw_cats.setdefault(key, set()).add(val)
    for key, v in cats.items():
        raw = raw_cats[key]
        if len(v) != len(raw):
            conflicts_by_value = OrderedDict()
            for raw_val in raw_cats[key]:
                conflicts_by_value.setdefault(val_to_num(raw_val), set()).add(raw_val)
            conflicts = [c for k in conflicts_by_value.values() if len(k) > 1 for c in k]
            raise ValueError('Partition names map to the same value: %s' % conflicts)
        vals_by_type = groupby_types(v)
        if len(vals_by_type) > 1:
            examples = [x[0] for x in vals_by_type.values()]
            warnings.warn('Partition names coerce to values of different types, e.g. %s' % examples)
    cats = OrderedDict([(key, list(v)) for key, v in cats.items()])
    return cats