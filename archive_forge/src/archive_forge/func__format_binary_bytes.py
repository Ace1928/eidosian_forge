import traceback
from collections import deque, namedtuple
from decimal import Decimal
from itertools import chain
from numbers import Number
from pprint import _recursion
from typing import Any, AnyStr, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple  # noqa
from .text import truncate
def _format_binary_bytes(val, maxlen, ellipsis='...'):
    if maxlen and len(val) > maxlen:
        chunk = memoryview(val)[:maxlen].tobytes()
        return _bytes_prefix(f"'{_repr_binary_bytes(chunk)}{ellipsis}'")
    return _bytes_prefix(f"'{_repr_binary_bytes(val)}'")