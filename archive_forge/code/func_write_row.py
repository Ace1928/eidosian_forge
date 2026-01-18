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
def write_row(self, row: pa.Table, writer_batch_size: Optional[int]=None):
    """Add a given single-row Table to the write-pool of rows which is written to file.

        Args:
            row: the row to add.
        """
    if len(row) != 1:
        raise ValueError(f'Only single-row pyarrow tables are allowed but got table with {len(row)} rows.')
    self.current_rows.append(row)
    if writer_batch_size is None:
        writer_batch_size = self.writer_batch_size
    if writer_batch_size is not None and len(self.current_rows) >= writer_batch_size:
        self.write_rows_on_file()