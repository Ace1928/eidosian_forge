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
def _cat_common(self, path, start=None, end=None):
    path = self._strip_protocol(path)
    logger.debug(f'cat: {path}')
    try:
        part = self.references[path]
    except KeyError:
        raise FileNotFoundError(path)
    if isinstance(part, str):
        part = part.encode()
    if isinstance(part, bytes):
        logger.debug(f'Reference: {path}, type bytes')
        if part.startswith(b'base64:'):
            part = base64.b64decode(part[7:])
        return (part, None, None)
    if len(part) == 1:
        logger.debug(f'Reference: {path}, whole file => {part}')
        url = part[0]
        start1, end1 = (start, end)
    else:
        url, start0, size = part
        logger.debug(f'Reference: {path} => {url}, offset {start0}, size {size}')
        end0 = start0 + size
        if start is not None:
            if start >= 0:
                start1 = start0 + start
            else:
                start1 = end0 + start
        else:
            start1 = start0
        if end is not None:
            if end >= 0:
                end1 = start0 + end
            else:
                end1 = end0 + end
        else:
            end1 = end0
    if url is None:
        url = self.target
    return (url, start1, end1)