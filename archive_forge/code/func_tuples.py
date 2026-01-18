from __future__ import annotations
import operator
from typing import Any
from typing import AsyncIterator
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from . import exc as async_exc
from ... import util
from ...engine import Result
from ...engine.result import _NO_ROW
from ...engine.result import _R
from ...engine.result import _WithKeys
from ...engine.result import FilterResult
from ...engine.result import FrozenResult
from ...engine.result import ResultMetaData
from ...engine.row import Row
from ...engine.row import RowMapping
from ...sql.base import _generative
from ...util.concurrency import greenlet_spawn
from ...util.typing import Literal
from ...util.typing import Self
def tuples(self) -> AsyncTupleResult[_TP]:
    """Apply a "typed tuple" typing filter to returned rows.

        This method returns the same :class:`_asyncio.AsyncResult` object
        at runtime,
        however annotates as returning a :class:`_asyncio.AsyncTupleResult`
        object that will indicate to :pep:`484` typing tools that plain typed
        ``Tuple`` instances are returned rather than rows.  This allows
        tuple unpacking and ``__getitem__`` access of :class:`_engine.Row`
        objects to by typed, for those cases where the statement invoked
        itself included typing information.

        .. versionadded:: 2.0

        :return: the :class:`_result.AsyncTupleResult` type at typing time.

        .. seealso::

            :attr:`_asyncio.AsyncResult.t` - shorter synonym

            :attr:`_engine.Row.t` - :class:`_engine.Row` version

        """
    return self