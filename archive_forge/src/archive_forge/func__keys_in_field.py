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
def _keys_in_field(self, field):
    """List key names in given field

        Produces strings like "field/x.y" appropriate from the chunking of the array
        """
    chunk_sizes = self._get_chunk_sizes(field)
    if len(chunk_sizes) == 0:
        yield (field + '/0')
        return
    inds = itertools.product(*(range(i) for i in chunk_sizes))
    for ind in inds:
        yield (field + '/' + '.'.join([str(c) for c in ind]))