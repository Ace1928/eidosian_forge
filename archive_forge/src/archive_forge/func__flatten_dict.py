import inspect
import tempfile
from collections import OrderedDict, UserDict
from collections.abc import MutableMapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial
from typing import Any, ContextManager, Iterable, List, Tuple
import numpy as np
from packaging import version
from .import_utils import get_torch_version, is_flax_available, is_tf_available, is_torch_available, is_torch_fx_proxy
def _flatten_dict(d, parent_key='', delimiter='.'):
    for k, v in d.items():
        key = str(parent_key) + delimiter + str(k) if parent_key else k
        if v and isinstance(v, MutableMapping):
            yield from flatten_dict(v, key, delimiter=delimiter).items()
        else:
            yield (key, v)