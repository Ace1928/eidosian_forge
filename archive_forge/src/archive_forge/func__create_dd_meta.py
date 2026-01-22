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
@classmethod
def _create_dd_meta(cls, dataset_info):
    pf = dataset_info['pf']
    index = dataset_info['index']
    categories = dataset_info['categories']
    columns = None
    pandas_md = pf.pandas_metadata
    if pandas_md:
        index_names, column_names, storage_name_mapping, column_index_names = _parse_pandas_metadata(pandas_md)
        column_names.extend(pf.cats)
    else:
        index_names = []
        column_names = pf.columns + list(pf.cats)
        storage_name_mapping = {k: k for k in column_names}
        column_index_names = [None]
    if index is None and len(index_names) > 0:
        if len(index_names) == 1 and index_names[0] is not None:
            index = index_names[0]
        else:
            index = index_names
    column_names, index_names = _normalize_index_columns(columns, column_names, index, index_names)
    all_columns = index_names + column_names
    categories_dict = None
    if isinstance(categories, dict):
        categories_dict = categories
    if categories is None:
        categories = pf.categories
    elif isinstance(categories, str):
        categories = [categories]
    else:
        categories = list(categories)
    if categories and (not set(categories).intersection(all_columns)):
        raise ValueError('categories not in available columns.\ncategories: {} | columns: {}'.format(categories, list(all_columns)))
    dtypes = pf._dtypes(categories)
    dtypes = {storage_name_mapping.get(k, k): v for k, v in dtypes.items()}
    index_cols = index or ()
    if isinstance(index_cols, str):
        index_cols = [index_cols]
    for ind in index_cols:
        if getattr(dtypes.get(ind), 'numpy_dtype', None):
            dtypes[ind] = dtypes[ind].numpy_dtype
    for cat in categories:
        if cat in all_columns:
            dtypes[cat] = pd.CategoricalDtype(categories=[UNKNOWN_CATEGORIES])
    for catcol in pf.cats:
        if catcol in all_columns:
            dtypes[catcol] = pd.CategoricalDtype(categories=pf.cats[catcol])
    meta = _meta_from_dtypes(all_columns, dtypes, index_cols, column_index_names)
    dataset_info['dtypes'] = dtypes
    dataset_info['index'] = index
    dataset_info['index_cols'] = index_cols
    dataset_info['categories'] = categories
    dataset_info['categories_dict'] = categories_dict
    return meta