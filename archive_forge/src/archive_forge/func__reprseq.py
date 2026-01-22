import traceback
from collections import deque, namedtuple
from decimal import Decimal
from itertools import chain
from numbers import Number
from pprint import _recursion
from typing import Any, AnyStr, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple  # noqa
from .text import truncate
def _reprseq(val, lit_start, lit_end, builtin_type, chainer):
    if type(val) is builtin_type:
        return (lit_start, lit_end, chainer(val))
    return (_literal(f'{type(val).__name__}({lit_start.value}', False, +1), _literal(f'{lit_end.value})', False, -1), chainer(val))