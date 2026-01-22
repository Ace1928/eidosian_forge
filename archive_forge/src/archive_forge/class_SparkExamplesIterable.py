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
class SparkExamplesIterable(_BaseExamplesIterable):

    def __init__(self, df: 'pyspark.sql.DataFrame', partition_order=None):
        self.df = df
        self.partition_order = partition_order or range(self.df.rdd.getNumPartitions())
        self.generate_examples_fn = _generate_iterable_examples(self.df, self.partition_order)

    def __iter__(self):
        yield from self.generate_examples_fn()

    def shuffle_data_sources(self, generator: np.random.Generator) -> 'SparkExamplesIterable':
        partition_order = list(range(self.df.rdd.getNumPartitions()))
        generator.shuffle(partition_order)
        return SparkExamplesIterable(self.df, partition_order=partition_order)

    def shard_data_sources(self, worker_id: int, num_workers: int) -> 'SparkExamplesIterable':
        partition_order = self.split_shard_indices_by_worker(worker_id, num_workers)
        return SparkExamplesIterable(self.df, partition_order=partition_order)

    @property
    def n_shards(self) -> int:
        return len(self.partition_order)