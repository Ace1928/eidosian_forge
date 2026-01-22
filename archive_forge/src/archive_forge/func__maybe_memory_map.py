from __future__ import annotations
from abc import (
import codecs
from collections import defaultdict
from collections.abc import (
import dataclasses
import functools
import gzip
from io import (
import mmap
import os
from pathlib import Path
import re
import tarfile
from typing import (
from urllib.parse import (
import warnings
import zipfile
from pandas._typing import (
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import ABCMultiIndex
from pandas.core.shared_docs import _shared_docs
def _maybe_memory_map(handle: str | BaseBuffer, memory_map: bool) -> tuple[str | BaseBuffer, bool, list[BaseBuffer]]:
    """Try to memory map file/buffer."""
    handles: list[BaseBuffer] = []
    memory_map &= hasattr(handle, 'fileno') or isinstance(handle, str)
    if not memory_map:
        return (handle, memory_map, handles)
    handle = cast(ReadCsvBuffer, handle)
    if isinstance(handle, str):
        handle = open(handle, 'rb')
        handles.append(handle)
    try:
        wrapped = _IOWrapper(mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ))
    finally:
        for handle in reversed(handles):
            handle.close()
    return (wrapped, memory_map, [wrapped])