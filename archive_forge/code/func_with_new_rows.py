from __future__ import annotations
from enum import Enum
import functools
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .row import Row
from .row import RowMapping
from .. import exc
from .. import util
from ..sql.base import _generative
from ..sql.base import HasMemoized
from ..sql.base import InPlaceGenerative
from ..util import HasMemoized_ro_memoized_attribute
from ..util import NONE_SET
from ..util._has_cy import HAS_CYEXTENSION
from ..util.typing import Literal
from ..util.typing import Self
def with_new_rows(self, tuple_data: Sequence[Row[_TP]]) -> FrozenResult[_TP]:
    fr = FrozenResult.__new__(FrozenResult)
    fr.metadata = self.metadata
    fr._attributes = self._attributes
    fr._source_supports_scalars = self._source_supports_scalars
    if self._source_supports_scalars:
        fr.data = [d[0] for d in tuple_data]
    else:
        fr.data = tuple_data
    return fr