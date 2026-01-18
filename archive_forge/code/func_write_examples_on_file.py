import errno
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from . import config
from .features import Features, Image, Value
from .features.features import (
from .filesystems import is_remote_filesystem
from .info import DatasetInfo
from .keyhash import DuplicatedKeysError, KeyHasher
from .table import array_cast, cast_array_to_feature, embed_table_storage, table_cast
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.file_utils import hash_url_to_filename
from .utils.py_utils import asdict, first_non_null_value
def write_examples_on_file(self):
    """Write stored examples from the write-pool of examples. It makes a table out of the examples and write it."""
    if not self.current_examples:
        return
    if self.schema:
        schema_cols = set(self.schema.names)
        examples_cols = self.current_examples[0][0].keys()
        common_cols = [col for col in self.schema.names if col in examples_cols]
        extra_cols = [col for col in examples_cols if col not in schema_cols]
        cols = common_cols + extra_cols
    else:
        cols = list(self.current_examples[0][0])
    batch_examples = {}
    for col in cols:
        if all((isinstance(row[0][col], (pa.Array, pa.ChunkedArray)) for row in self.current_examples)):
            arrays = [row[0][col] for row in self.current_examples]
            arrays = [chunk for array in arrays for chunk in (array.chunks if isinstance(array, pa.ChunkedArray) else [array])]
            batch_examples[col] = pa.concat_arrays(arrays)
        else:
            batch_examples[col] = [row[0][col].to_pylist()[0] if isinstance(row[0][col], (pa.Array, pa.ChunkedArray)) else row[0][col] for row in self.current_examples]
    self.write_batch(batch_examples=batch_examples)
    self.current_examples = []