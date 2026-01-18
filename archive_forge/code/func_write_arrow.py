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
def write_arrow(it):
    task_id = pyspark.TaskContext().taskAttemptId()
    first_batch = next(it, None)
    if first_batch is None:
        return pa.RecordBatch.from_arrays([[task_id], [0], [0]], names=['task_id', 'num_examples', 'num_bytes'])
    shard_id = 0
    writer = writer_class(features=features, path=working_fpath.replace('SSSSS', f'{shard_id:05d}').replace('TTTTT', f'{task_id:05d}'), writer_batch_size=writer_batch_size, storage_options=storage_options, embed_local_files=embed_local_files)
    table = pa.Table.from_batches([first_batch])
    writer.write_table(table)
    for batch in it:
        if max_shard_size is not None and writer._num_bytes >= max_shard_size:
            num_examples, num_bytes = writer.finalize()
            writer.close()
            yield pa.RecordBatch.from_arrays([[task_id], [num_examples], [num_bytes]], names=['task_id', 'num_examples', 'num_bytes'])
            shard_id += 1
            writer = writer_class(features=writer._features, path=working_fpath.replace('SSSSS', f'{shard_id:05d}').replace('TTTTT', f'{task_id:05d}'), writer_batch_size=writer_batch_size, storage_options=storage_options, embed_local_files=embed_local_files)
        table = pa.Table.from_batches([batch])
        writer.write_table(table)
    if writer._num_bytes > 0:
        num_examples, num_bytes = writer.finalize()
        writer.close()
        yield pa.RecordBatch.from_arrays([[task_id], [num_examples], [num_bytes]], names=['task_id', 'num_examples', 'num_bytes'])
    if working_fpath != fpath:
        for file in os.listdir(os.path.dirname(working_fpath)):
            dest = os.path.join(os.path.dirname(fpath), os.path.basename(file))
            shutil.move(file, dest)