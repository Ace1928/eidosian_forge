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
def _deserialize_column_index(block_table, all_columns, column_indexes):
    column_strings = [frombytes(x) if isinstance(x, bytes) else x for x in block_table.column_names]
    if all_columns:
        columns_name_dict = {c.get('field_name', _column_name_to_strings(c['name'])): c['name'] for c in all_columns}
        columns_values = [columns_name_dict.get(name, name) for name in column_strings]
    else:
        columns_values = column_strings
    to_pair = ast.literal_eval if len(column_indexes) > 1 else lambda x: (x,)
    if not columns_values:
        columns = _pandas_api.pd.Index(columns_values)
    else:
        columns = _pandas_api.pd.MultiIndex.from_tuples(list(map(to_pair, columns_values)), names=[col_index['name'] for col_index in column_indexes] or None)
    if len(column_indexes) > 0:
        columns = _reconstruct_columns_from_metadata(columns, column_indexes)
    columns = _flatten_single_level_multiindex(columns)
    return columns