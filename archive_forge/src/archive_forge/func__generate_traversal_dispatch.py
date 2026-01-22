from __future__ import annotations
from collections import deque
from enum import Enum
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import exc
from .. import util
from ..util import langhelpers
from ..util._has_cy import HAS_CYEXTENSION
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
def _generate_traversal_dispatch() -> None:
    lookup = _dispatch_lookup
    for sym in InternalTraversal:
        key = sym.name
        if key.startswith('dp_'):
            visit_key = key.replace('dp_', 'visit_')
            sym_name = sym.value
            assert sym_name not in lookup, sym_name
            lookup[sym] = lookup[sym_name] = visit_key