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
def write_batch(self, batch_examples: Dict[str, List], writer_batch_size: Optional[int]=None):
    """Write a batch of Example to file.
        Ignores the batch if it appears to be empty,
        preventing a potential schema update of unknown types.

        Args:
            batch_examples: the batch of examples to add.
        """
    if batch_examples and len(next(iter(batch_examples.values()))) == 0:
        return
    features = None if self.pa_writer is None and self.update_features else self._features
    try_features = self._features if self.pa_writer is None and self.update_features else None
    arrays = []
    inferred_features = Features()
    if self.schema:
        schema_cols = set(self.schema.names)
        batch_cols = batch_examples.keys()
        common_cols = [col for col in self.schema.names if col in batch_cols]
        extra_cols = [col for col in batch_cols if col not in schema_cols]
        cols = common_cols + extra_cols
    else:
        cols = list(batch_examples)
    for col in cols:
        col_values = batch_examples[col]
        col_type = features[col] if features else None
        if isinstance(col_values, (pa.Array, pa.ChunkedArray)):
            array = cast_array_to_feature(col_values, col_type) if col_type is not None else col_values
            arrays.append(array)
            inferred_features[col] = generate_from_arrow_type(col_values.type)
        else:
            col_try_type = try_features[col] if try_features is not None and col in try_features else None
            typed_sequence = OptimizedTypedSequence(col_values, type=col_type, try_type=col_try_type, col=col)
            arrays.append(pa.array(typed_sequence))
            inferred_features[col] = typed_sequence.get_inferred_type()
    schema = inferred_features.arrow_schema if self.pa_writer is None else self.schema
    pa_table = pa.Table.from_arrays(arrays, schema=schema)
    self.write_table(pa_table, writer_batch_size)