import traceback
from collections import deque, namedtuple
from decimal import Decimal
from itertools import chain
from numbers import Number
from pprint import _recursion
from typing import Any, AnyStr, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple  # noqa
from .text import truncate
def _chaindict(mapping, LIT_DICT_KVSEP=LIT_DICT_KVSEP, LIT_LIST_SEP=LIT_LIST_SEP):
    size = len(mapping)
    for i, (k, v) in enumerate(mapping.items()):
        yield _key(k)
        yield LIT_DICT_KVSEP
        yield v
        if i < size - 1:
            yield LIT_LIST_SEP