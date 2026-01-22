import itertools
import logging
import os
import pathlib
import re
from typing import (
import numpy as np
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import BlockMetadata
from ray.data.datasource.partitioning import Partitioning
from ray.util.annotations import DeveloperAPI
def _get_block_metadata(self, paths: List[str], schema: Optional[Union[type, 'pyarrow.lib.Schema']], *, num_fragments: int, prefetched_metadata: Optional[List['pyarrow.parquet.FileMetaData']]) -> BlockMetadata:
    if prefetched_metadata is not None and len(prefetched_metadata) == num_fragments and all((m is not None for m in prefetched_metadata)):
        block_metadata = BlockMetadata(num_rows=sum((m.num_rows for m in prefetched_metadata)), size_bytes=sum((sum((m.row_group(i).total_byte_size for i in range(m.num_row_groups))) for m in prefetched_metadata)), schema=schema, input_files=paths, exec_stats=None)
    else:
        block_metadata = BlockMetadata(num_rows=None, size_bytes=None, schema=schema, input_files=paths, exec_stats=None)
    return block_metadata