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
@lru_cache(4096)
def _key_to_record(self, key):
    """Details needed to construct a reference for one key"""
    field, chunk = key.rsplit('/', 1)
    chunk_sizes = self._get_chunk_sizes(field)
    if len(chunk_sizes) == 0:
        return (0, 0, 0)
    chunk_idx = [int(c) for c in chunk.split('.')]
    chunk_number = ravel_multi_index(chunk_idx, chunk_sizes)
    record = chunk_number // self.record_size
    ri = chunk_number % self.record_size
    return (record, ri, len(chunk_sizes))