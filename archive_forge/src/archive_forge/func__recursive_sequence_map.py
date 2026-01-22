from collections.abc import Collection
from collections.abc import Sized
from decimal import Decimal
import math
from numbers import Complex
import pprint
from types import TracebackType
from typing import Any
from typing import Callable
from typing import cast
from typing import ContextManager
from typing import final
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Pattern
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import _pytest._code
from _pytest.outcomes import fail
def _recursive_sequence_map(f, x):
    """Recursively map a function over a sequence of arbitrary depth"""
    if isinstance(x, (list, tuple)):
        seq_type = type(x)
        return seq_type((_recursive_sequence_map(f, xi) for xi in x))
    else:
        return f(x)