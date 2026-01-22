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
def _generate_record(self, field, record):
    """The references for a given parquet file of a given field"""
    refs = self.open_refs(field, record)
    it = iter(zip(*refs.values()))
    if len(refs) == 3:
        return (list(t) for t in it)
    elif len(refs) == 1:
        return refs['raw']
    else:
        return (list(t[:3]) if not t[3] else t[3] for t in it)