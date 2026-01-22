from __future__ import annotations
import asyncio
import contextlib
from typing import Any
from typing import AsyncIterator
from typing import Callable
from typing import Dict
from typing import Generator
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import exc as async_exc
from .base import asyncstartablecontext
from .base import GeneratorStartableContext
from .base import ProxyComparable
from .base import StartableContext
from .result import _ensure_sync_result
from .result import AsyncResult
from .result import AsyncScalarResult
from ... import exc
from ... import inspection
from ... import util
from ...engine import Connection
from ...engine import create_engine as _create_engine
from ...engine import create_pool_from_url as _create_pool_from_url
from ...engine import Engine
from ...engine.base import NestedTransaction
from ...engine.base import Transaction
from ...exc import ArgumentError
from ...util.concurrency import greenlet_spawn
from ...util.typing import Concatenate
from ...util.typing import ParamSpec
@util.create_proxy_methods(Engine, ':class:`_engine.Engine`', ':class:`_asyncio.AsyncEngine`', classmethods=[], methods=['clear_compiled_cache', 'update_execution_options', 'get_execution_options'], attributes=['url', 'pool', 'dialect', 'engine', 'name', 'driver', 'echo'])
class AsyncEngine(ProxyComparable[Engine], AsyncConnectable):
    """An asyncio proxy for a :class:`_engine.Engine`.

    :class:`_asyncio.AsyncEngine` is acquired using the
    :func:`_asyncio.create_async_engine` function::

        from sqlalchemy.ext.asyncio import create_async_engine
        engine = create_async_engine("postgresql+asyncpg://user:pass@host/dbname")

    .. versionadded:: 1.4

    """
    __slots__ = 'sync_engine'
    _connection_cls: Type[AsyncConnection] = AsyncConnection
    sync_engine: Engine
    'Reference to the sync-style :class:`_engine.Engine` this\n    :class:`_asyncio.AsyncEngine` proxies requests towards.\n\n    This instance can be used as an event target.\n\n    .. seealso::\n\n        :ref:`asyncio_events`\n    '

    def __init__(self, sync_engine: Engine):
        if not sync_engine.dialect.is_async:
            raise exc.InvalidRequestError(f'The asyncio extension requires an async driver to be used. The loaded {sync_engine.dialect.driver!r} is not async.')
        self.sync_engine = self._assign_proxied(sync_engine)

    @util.ro_non_memoized_property
    def _proxied(self) -> Engine:
        return self.sync_engine

    @classmethod
    def _regenerate_proxy_for_target(cls, target: Engine) -> AsyncEngine:
        return AsyncEngine(target)

    @contextlib.asynccontextmanager
    async def begin(self) -> AsyncIterator[AsyncConnection]:
        """Return a context manager which when entered will deliver an
        :class:`_asyncio.AsyncConnection` with an
        :class:`_asyncio.AsyncTransaction` established.

        E.g.::

            async with async_engine.begin() as conn:
                await conn.execute(
                    text("insert into table (x, y, z) values (1, 2, 3)")
                )
                await conn.execute(text("my_special_procedure(5)"))


        """
        conn = self.connect()
        async with conn:
            async with conn.begin():
                yield conn

    def connect(self) -> AsyncConnection:
        """Return an :class:`_asyncio.AsyncConnection` object.

        The :class:`_asyncio.AsyncConnection` will procure a database
        connection from the underlying connection pool when it is entered
        as an async context manager::

            async with async_engine.connect() as conn:
                result = await conn.execute(select(user_table))

        The :class:`_asyncio.AsyncConnection` may also be started outside of a
        context manager by invoking its :meth:`_asyncio.AsyncConnection.start`
        method.

        """
        return self._connection_cls(self)

    async def raw_connection(self) -> PoolProxiedConnection:
        """Return a "raw" DBAPI connection from the connection pool.

        .. seealso::

            :ref:`dbapi_connections`

        """
        return await greenlet_spawn(self.sync_engine.raw_connection)

    @overload
    def execution_options(self, *, compiled_cache: Optional[CompiledCacheType]=..., logging_token: str=..., isolation_level: IsolationLevel=..., insertmanyvalues_page_size: int=..., schema_translate_map: Optional[SchemaTranslateMapType]=..., **opt: Any) -> AsyncEngine:
        ...

    @overload
    def execution_options(self, **opt: Any) -> AsyncEngine:
        ...

    def execution_options(self, **opt: Any) -> AsyncEngine:
        """Return a new :class:`_asyncio.AsyncEngine` that will provide
        :class:`_asyncio.AsyncConnection` objects with the given execution
        options.

        Proxied from :meth:`_engine.Engine.execution_options`.  See that
        method for details.

        """
        return AsyncEngine(self.sync_engine.execution_options(**opt))

    async def dispose(self, close: bool=True) -> None:
        """Dispose of the connection pool used by this
        :class:`_asyncio.AsyncEngine`.

        :param close: if left at its default of ``True``, has the
         effect of fully closing all **currently checked in**
         database connections.  Connections that are still checked out
         will **not** be closed, however they will no longer be associated
         with this :class:`_engine.Engine`,
         so when they are closed individually, eventually the
         :class:`_pool.Pool` which they are associated with will
         be garbage collected and they will be closed out fully, if
         not already closed on checkin.

         If set to ``False``, the previous connection pool is de-referenced,
         and otherwise not touched in any way.

        .. seealso::

            :meth:`_engine.Engine.dispose`

        """
        await greenlet_spawn(self.sync_engine.dispose, close=close)

    def clear_compiled_cache(self) -> None:
        """Clear the compiled cache associated with the dialect.

        .. container:: class_bases

            Proxied for the :class:`_engine.Engine` class on
            behalf of the :class:`_asyncio.AsyncEngine` class.

        This applies **only** to the built-in cache that is established
        via the :paramref:`_engine.create_engine.query_cache_size` parameter.
        It will not impact any dictionary caches that were passed via the
        :paramref:`.Connection.execution_options.compiled_cache` parameter.

        .. versionadded:: 1.4


        """
        return self._proxied.clear_compiled_cache()

    def update_execution_options(self, **opt: Any) -> None:
        """Update the default execution_options dictionary
        of this :class:`_engine.Engine`.

        .. container:: class_bases

            Proxied for the :class:`_engine.Engine` class on
            behalf of the :class:`_asyncio.AsyncEngine` class.

        The given keys/values in \\**opt are added to the
        default execution options that will be used for
        all connections.  The initial contents of this dictionary
        can be sent via the ``execution_options`` parameter
        to :func:`_sa.create_engine`.

        .. seealso::

            :meth:`_engine.Connection.execution_options`

            :meth:`_engine.Engine.execution_options`


        """
        return self._proxied.update_execution_options(**opt)

    def get_execution_options(self) -> _ExecuteOptions:
        """Get the non-SQL options which will take effect during execution.

        .. container:: class_bases

            Proxied for the :class:`_engine.Engine` class on
            behalf of the :class:`_asyncio.AsyncEngine` class.

        .. versionadded: 1.3

        .. seealso::

            :meth:`_engine.Engine.execution_options`

        """
        return self._proxied.get_execution_options()

    @property
    def url(self) -> URL:
        """Proxy for the :attr:`_engine.Engine.url` attribute
        on behalf of the :class:`_asyncio.AsyncEngine` class.

        """
        return self._proxied.url

    @url.setter
    def url(self, attr: URL) -> None:
        self._proxied.url = attr

    @property
    def pool(self) -> Pool:
        """Proxy for the :attr:`_engine.Engine.pool` attribute
        on behalf of the :class:`_asyncio.AsyncEngine` class.

        """
        return self._proxied.pool

    @pool.setter
    def pool(self, attr: Pool) -> None:
        self._proxied.pool = attr

    @property
    def dialect(self) -> Dialect:
        """Proxy for the :attr:`_engine.Engine.dialect` attribute
        on behalf of the :class:`_asyncio.AsyncEngine` class.

        """
        return self._proxied.dialect

    @dialect.setter
    def dialect(self, attr: Dialect) -> None:
        self._proxied.dialect = attr

    @property
    def engine(self) -> Any:
        """Returns this :class:`.Engine`.

        .. container:: class_bases

            Proxied for the :class:`_engine.Engine` class
            on behalf of the :class:`_asyncio.AsyncEngine` class.

        Used for legacy schemes that accept :class:`.Connection` /
        :class:`.Engine` objects within the same variable.


        """
        return self._proxied.engine

    @property
    def name(self) -> Any:
        """String name of the :class:`~sqlalchemy.engine.interfaces.Dialect`
        in use by this :class:`Engine`.

        .. container:: class_bases

            Proxied for the :class:`_engine.Engine` class
            on behalf of the :class:`_asyncio.AsyncEngine` class.


        """
        return self._proxied.name

    @property
    def driver(self) -> Any:
        """Driver name of the :class:`~sqlalchemy.engine.interfaces.Dialect`
        in use by this :class:`Engine`.

        .. container:: class_bases

            Proxied for the :class:`_engine.Engine` class
            on behalf of the :class:`_asyncio.AsyncEngine` class.


        """
        return self._proxied.driver

    @property
    def echo(self) -> Any:
        """When ``True``, enable log output for this element.

        .. container:: class_bases

            Proxied for the :class:`_engine.Engine` class
            on behalf of the :class:`_asyncio.AsyncEngine` class.

        This has the effect of setting the Python logging level for the namespace
        of this element's class and object reference.  A value of boolean ``True``
        indicates that the loglevel ``logging.INFO`` will be set for the logger,
        whereas the string value ``debug`` will set the loglevel to
        ``logging.DEBUG``.

        """
        return self._proxied.echo

    @echo.setter
    def echo(self, attr: Any) -> None:
        self._proxied.echo = attr