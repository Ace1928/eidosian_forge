import traceback
from collections import deque, namedtuple
from decimal import Decimal
from itertools import chain
from numbers import Number
from pprint import _recursion
from typing import Any, AnyStr, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple  # noqa
from .text import truncate
def _repr_binary_bytes(val):
    try:
        return val.decode('utf-8')
    except UnicodeDecodeError:
        return val.hex()