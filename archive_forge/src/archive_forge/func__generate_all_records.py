import base64
import collections
import io
import itertools
import logging
import math
import os
from functools import lru_cache
from typing import TYPE_CHECKING
import fsspec.core
from ..asyn import AsyncFileSystem
from ..callbacks import DEFAULT_CALLBACK
from ..core import filesystem, open, split_protocol
from ..utils import isfilelike, merge_offset_ranges, other_paths
def _generate_all_records(self, field):
    """Load all the references within a field by iterating over the parquet files"""
    nrec = 1
    for ch in self._get_chunk_sizes(field):
        nrec *= ch
    nrec = math.ceil(nrec / self.record_size)
    for record in range(nrec):
        yield from self._generate_record(field, record)