import collections.abc
import gc
import inspect
import re
import sys
import weakref
from functools import partial, wraps
from itertools import chain
from typing import (
from scrapy.utils.asyncgen import as_async_generator
def without_none_values(iterable: Union[Mapping, Iterable]) -> Union[dict, Iterable]:
    """Return a copy of ``iterable`` with all ``None`` entries removed.

    If ``iterable`` is a mapping, return a dictionary where all pairs that have
    value ``None`` have been removed.
    """
    if isinstance(iterable, collections.abc.Mapping):
        return {k: v for k, v in iterable.items() if v is not None}
    else:
        return type(iterable)((v for v in iterable if v is not None))