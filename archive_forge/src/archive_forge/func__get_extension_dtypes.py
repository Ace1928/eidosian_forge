import ast
from collections.abc import Sequence
from concurrent import futures
import concurrent.futures.thread  # noqa
from copy import deepcopy
from itertools import zip_longest
import json
import operator
import re
import warnings
import numpy as np
import pyarrow as pa
from pyarrow.lib import _pandas_api, frombytes  # noqa
def _get_extension_dtypes(table, columns_metadata, types_mapper=None):
    """
    Based on the stored column pandas metadata and the extension types
    in the arrow schema, infer which columns should be converted to a
    pandas extension dtype.

    The 'numpy_type' field in the column metadata stores the string
    representation of the original pandas dtype (and, despite its name,
    not the 'pandas_type' field).
    Based on this string representation, a pandas/numpy dtype is constructed
    and then we can check if this dtype supports conversion from arrow.

    """
    ext_columns = {}
    if _pandas_api.extension_dtype is None:
        return ext_columns
    for col_meta in columns_metadata:
        try:
            name = col_meta['field_name']
        except KeyError:
            name = col_meta['name']
        dtype = col_meta['numpy_type']
        if dtype not in _pandas_supported_numpy_types:
            pandas_dtype = _pandas_api.pandas_dtype(dtype)
            if isinstance(pandas_dtype, _pandas_api.extension_dtype):
                if hasattr(pandas_dtype, '__from_arrow__'):
                    ext_columns[name] = pandas_dtype
    for field in table.schema:
        typ = field.type
        if isinstance(typ, pa.BaseExtensionType):
            try:
                pandas_dtype = typ.to_pandas_dtype()
            except NotImplementedError:
                pass
            else:
                ext_columns[field.name] = pandas_dtype
    if types_mapper:
        for field in table.schema:
            typ = field.type
            pandas_dtype = types_mapper(typ)
            if pandas_dtype is not None:
                ext_columns[field.name] = pandas_dtype
    return ext_columns