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
def _flatten_single_level_multiindex(index):
    pd = _pandas_api.pd
    if isinstance(index, pd.MultiIndex) and index.nlevels == 1:
        levels, = index.levels
        labels, = _get_multiindex_codes(index)
        dtype = levels.dtype
        if not index.is_unique:
            raise ValueError('Found non-unique column index')
        return pd.Index([levels[_label] if _label != -1 else None for _label in labels], dtype=dtype, name=index.names[0])
    return index