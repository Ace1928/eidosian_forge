from collections import defaultdict
from contextlib import nullcontext
from functools import reduce
import inspect
import json
import os
import re
import operator
import warnings
import pyarrow as pa
from pyarrow._parquet import (ParquetReader, Statistics,  # noqa
from pyarrow.fs import (LocalFileSystem, FileSystem, FileType,
from pyarrow import filesystem as legacyfs
from pyarrow.util import guid, _is_path_like, _stringify_path, _deprecate_api
def _check_filters(filters, check_null_strings=True):
    """
    Check if filters are well-formed.
    """
    if filters is not None:
        if len(filters) == 0 or any((len(f) == 0 for f in filters)):
            raise ValueError('Malformed filters')
        if isinstance(filters[0][0], str):
            filters = [filters]
        if check_null_strings:
            for conjunction in filters:
                for col, op, val in conjunction:
                    if isinstance(val, list) and all((_check_contains_null(v) for v in val)) or _check_contains_null(val):
                        raise NotImplementedError('Null-terminated binary strings are not supported as filter values.')
    return filters