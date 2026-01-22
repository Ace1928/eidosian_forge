from __future__ import annotations
from collections import deque
import collections.abc as collections_abc
import itertools
from itertools import zip_longest
import operator
import typing
from typing import Any
from typing import Callable
from typing import Deque
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from . import operators
from .cache_key import HasCacheKey
from .visitors import _TraverseInternalsType
from .visitors import anon_map
from .visitors import ExternallyTraversible
from .visitors import HasTraversalDispatch
from .visitors import HasTraverseInternals
from .. import util
from ..util import langhelpers
from ..util.typing import Self
@classmethod
def _generate_shallow_copy(cls, internal_dispatch: _TraverseInternalsType, method_name: str) -> Callable[[Self, Self], None]:
    code = '\n'.join((f'    other.{attrname} = self.{attrname}' for attrname, _ in internal_dispatch))
    meth_text = f'def {method_name}(self, other):\n{code}\n'
    return langhelpers._exec_code_in_env(meth_text, {}, method_name)