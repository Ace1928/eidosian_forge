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
def _reorder_dataframe_by_partition(df: 'pyspark.sql.DataFrame', new_partition_order: List[int]):
    df_combined = df.select('*').where(f'part_id = {new_partition_order[0]}')
    for partition_id in new_partition_order[1:]:
        partition_df = df.select('*').where(f'part_id = {partition_id}')
        df_combined = df_combined.union(partition_df)
    return df_combined