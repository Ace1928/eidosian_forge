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
def _get_simple_index_descriptor(level, name):
    string_dtype, extra_metadata = get_extension_dtype_info(level)
    pandas_type = get_logical_type_from_numpy(level)
    if 'mixed' in pandas_type:
        warnings.warn('The DataFrame has column names of mixed type. They will be converted to strings and not roundtrip correctly.', UserWarning, stacklevel=4)
    if pandas_type == 'unicode':
        assert not extra_metadata
        extra_metadata = {'encoding': 'UTF-8'}
    return {'name': name, 'field_name': name, 'pandas_type': pandas_type, 'numpy_type': string_dtype, 'metadata': extra_metadata}