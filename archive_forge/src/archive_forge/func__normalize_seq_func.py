from __future__ import annotations
import dataclasses
import datetime
import decimal
import hashlib
import inspect
import pathlib
import pickle
import types
import uuid
import warnings
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Iterator, Mapping
from concurrent.futures import Executor
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from functools import partial
from numbers import Integral, Number
from operator import getitem
from typing import Any, Literal, TypeVar
import cloudpickle
from tlz import curry, groupby, identity, merge
from tlz.functoolz import Compose
from dask import config, local
from dask._compatibility import EMSCRIPTEN
from dask.core import flatten
from dask.core import get as simple_get
from dask.core import literal, quote
from dask.hashing import hash_buffer_hex
from dask.system import CPU_COUNT
from dask.typing import Key, SchedulerGetCallable
from dask.utils import (
def _normalize_seq_func(seq: Iterable[object]) -> list[object]:
    with _seen_ctx(reset=False) as seen:
        out = []
        for item in seq:
            if isinstance(item, (str, bytes, int, float, bool, type(None))):
                pass
            elif id(item) in seen:
                seen_when, _ = seen[id(item)]
                item = ('__seen', seen_when)
            else:
                seen[id(item)] = (len(seen), item)
                item = normalize_token(item)
            out.append(item)
        return out