from __future__ import annotations
import abc
import copy
import html
from collections.abc import (
from typing import Any
import tlz as toolz
import dask
from dask import config
from dask.base import clone_key, flatten, is_dask_collection, normalize_token
from dask.core import keys_in_tasks, reverse_dict
from dask.typing import DaskCollection, Graph, Key
from dask.utils import ensure_dict, import_required, key_split
from dask.widgets import get_template
def _get_some_layer_name(collection) -> str:
    """Somehow get a unique name for a Layer from a non-HighLevelGraph dask mapping"""
    try:
        name, = collection.__dask_layers__()
        return name
    except (AttributeError, ValueError):
        return str(id(collection))