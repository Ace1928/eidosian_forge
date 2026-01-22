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
def _build_nested_paths(self):
    paths = self.reader.column_paths
    result = defaultdict(list)
    for i, path in enumerate(paths):
        key = path[0]
        rest = path[1:]
        while True:
            result[key].append(i)
            if not rest:
                break
            key = '.'.join((key, rest[0]))
            rest = rest[1:]
    return result