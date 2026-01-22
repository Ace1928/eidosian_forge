from __future__ import annotations
from collections import deque
import dataclasses
from enum import Enum
import threading
import time
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import weakref
from .. import event
from .. import exc
from .. import log
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
class ManagesConnection:
    """Common base for the two connection-management interfaces
    :class:`.PoolProxiedConnection` and :class:`.ConnectionPoolEntry`.

    These two objects are typically exposed in the public facing API
    via the connection pool event hooks, documented at :class:`.PoolEvents`.

    .. versionadded:: 2.0

    """
    __slots__ = ()
    dbapi_connection: Optional[DBAPIConnection]
    "A reference to the actual DBAPI connection being tracked.\n\n    This is a :pep:`249`-compliant object that for traditional sync-style\n    dialects is provided by the third-party\n    DBAPI implementation in use.  For asyncio dialects, the implementation\n    is typically an adapter object provided by the SQLAlchemy dialect\n    itself; the underlying asyncio object is available via the\n    :attr:`.ManagesConnection.driver_connection` attribute.\n\n    SQLAlchemy's interface for the DBAPI connection is based on the\n    :class:`.DBAPIConnection` protocol object\n\n    .. seealso::\n\n        :attr:`.ManagesConnection.driver_connection`\n\n        :ref:`faq_dbapi_connection`\n\n    "
    driver_connection: Optional[Any]
    'The "driver level" connection object as used by the Python\n    DBAPI or database driver.\n\n    For traditional :pep:`249` DBAPI implementations, this object will\n    be the same object as that of\n    :attr:`.ManagesConnection.dbapi_connection`.   For an asyncio database\n    driver, this will be the ultimate "connection" object used by that\n    driver, such as the ``asyncpg.Connection`` object which will not have\n    standard pep-249 methods.\n\n    .. versionadded:: 1.4.24\n\n    .. seealso::\n\n        :attr:`.ManagesConnection.dbapi_connection`\n\n        :ref:`faq_dbapi_connection`\n\n    '

    @util.ro_memoized_property
    def info(self) -> _InfoType:
        """Info dictionary associated with the underlying DBAPI connection
        referred to by this :class:`.ManagesConnection` instance, allowing
        user-defined data to be associated with the connection.

        The data in this dictionary is persistent for the lifespan
        of the DBAPI connection itself, including across pool checkins
        and checkouts.  When the connection is invalidated
        and replaced with a new one, this dictionary is cleared.

        For a :class:`.PoolProxiedConnection` instance that's not associated
        with a :class:`.ConnectionPoolEntry`, such as if it were detached, the
        attribute returns a dictionary that is local to that
        :class:`.ConnectionPoolEntry`. Therefore the
        :attr:`.ManagesConnection.info` attribute will always provide a Python
        dictionary.

        .. seealso::

            :attr:`.ManagesConnection.record_info`


        """
        raise NotImplementedError()

    @util.ro_memoized_property
    def record_info(self) -> Optional[_InfoType]:
        """Persistent info dictionary associated with this
        :class:`.ManagesConnection`.

        Unlike the :attr:`.ManagesConnection.info` dictionary, the lifespan
        of this dictionary is that of the :class:`.ConnectionPoolEntry`
        which owns it; therefore this dictionary will persist across
        reconnects and connection invalidation for a particular entry
        in the connection pool.

        For a :class:`.PoolProxiedConnection` instance that's not associated
        with a :class:`.ConnectionPoolEntry`, such as if it were detached, the
        attribute returns None. Contrast to the :attr:`.ManagesConnection.info`
        dictionary which is never None.


        .. seealso::

            :attr:`.ManagesConnection.info`

        """
        raise NotImplementedError()

    def invalidate(self, e: Optional[BaseException]=None, soft: bool=False) -> None:
        """Mark the managed connection as invalidated.

        :param e: an exception object indicating a reason for the invalidation.

        :param soft: if True, the connection isn't closed; instead, this
         connection will be recycled on next checkout.

        .. seealso::

            :ref:`pool_connection_invalidation`


        """
        raise NotImplementedError()