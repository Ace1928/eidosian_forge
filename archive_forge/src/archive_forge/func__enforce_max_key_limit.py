from __future__ import annotations
import math
import numbers
from collections.abc import Iterable
from enum import Enum
from typing import Any
from dask import config, core, utils
from dask.base import normalize_token, tokenize
from dask.core import (
from dask.typing import Graph, Key
def _enforce_max_key_limit(key_name):
    if max_fused_key_length and len(key_name) > max_fused_key_length:
        name_hash = f'{hash(key_name):x}'[:4]
        key_name = f'{key_name[:max_fused_key_length]}-{name_hash}'
    return key_name