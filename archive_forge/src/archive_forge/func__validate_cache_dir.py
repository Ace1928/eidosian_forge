import os
import posixpath
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Union
import numpy as np
import pyarrow as pa
import datasets
from datasets.arrow_writer import ArrowWriter, ParquetWriter
from datasets.config import MAX_SHARD_SIZE
from datasets.filesystems import (
from datasets.iterable_dataset import _BaseExamplesIterable
from datasets.utils.py_utils import convert_file_size_to_int
def _validate_cache_dir(self):
    cache_dir = self._cache_dir

    def create_cache_and_write_probe(context):
        os.makedirs(cache_dir, exist_ok=True)
        probe_file = os.path.join(cache_dir, 'fs_test' + uuid.uuid4().hex)
        open(probe_file, 'a')
        return [probe_file]
    if self._spark.conf.get('spark.master', '').startswith('local'):
        return
    if self._cache_dir:
        probe = self._spark.sparkContext.parallelize(range(1), 1).mapPartitions(create_cache_and_write_probe).collect()
        if os.path.isfile(probe[0]):
            return
    raise ValueError('When using Dataset.from_spark on a multi-node cluster, the driver and all workers should be able to access cache_dir')