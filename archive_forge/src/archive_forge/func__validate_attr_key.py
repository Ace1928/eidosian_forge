from __future__ import annotations
import gzip
import io
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray.backends.common import (
from xarray.backends.file_manager import CachingFileManager, DummyFileManager
from xarray.backends.locks import ensure_lock, get_write_lock
from xarray.backends.netcdf3 import (
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import (
from xarray.core.variable import Variable
def _validate_attr_key(self, key):
    if not is_valid_nc3_name(key):
        raise ValueError('Not a valid attribute name')