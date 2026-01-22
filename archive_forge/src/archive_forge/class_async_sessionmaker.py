from __future__ import annotations
import asyncio
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import engine
from .base import ReversibleProxy
from .base import StartableContext
from .result import _ensure_sync_result
from .result import AsyncResult
from .result import AsyncScalarResult
from ... import util
from ...orm import close_all_sessions as _sync_close_all_sessions
from ...orm import object_session
from ...orm import Session
from ...orm import SessionTransaction
from ...orm import state as _instance_state
from ...util.concurrency import greenlet_spawn
from ...util.typing import Concatenate
from ...util.typing import ParamSpec
class async_sessionmaker(Generic[_AS]):
    """A configurable :class:`.AsyncSession` factory.

    The :class:`.async_sessionmaker` factory works in the same way as the
    :class:`.sessionmaker` factory, to generate new :class:`.AsyncSession`
    objects when called, creating them given
    the configurational arguments established here.

    e.g.::

        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.ext.asyncio import async_sessionmaker

        async def run_some_sql(async_session: async_sessionmaker[AsyncSession]) -> None:
            async with async_session() as session:
                session.add(SomeObject(data="object"))
                session.add(SomeOtherObject(name="other object"))
                await session.commit()

        async def main() -> None:
            # an AsyncEngine, which the AsyncSession will use for connection
            # resources
            engine = create_async_engine('postgresql+asyncpg://scott:tiger@localhost/')

            # create a reusable factory for new AsyncSession instances
            async_session = async_sessionmaker(engine)

            await run_some_sql(async_session)

            await engine.dispose()

    The :class:`.async_sessionmaker` is useful so that different parts
    of a program can create new :class:`.AsyncSession` objects with a
    fixed configuration established up front.  Note that :class:`.AsyncSession`
    objects may also be instantiated directly when not using
    :class:`.async_sessionmaker`.

    .. versionadded:: 2.0  :class:`.async_sessionmaker` provides a
       :class:`.sessionmaker` class that's dedicated to the
       :class:`.AsyncSession` object, including pep-484 typing support.

    .. seealso::

        :ref:`asyncio_orm` - shows example use

        :class:`.sessionmaker`  - general overview of the
         :class:`.sessionmaker` architecture


        :ref:`session_getting` - introductory text on creating
        sessions using :class:`.sessionmaker`.

    """
    class_: Type[_AS]

    @overload
    def __init__(self, bind: Optional[_AsyncSessionBind]=..., *, class_: Type[_AS], autoflush: bool=..., expire_on_commit: bool=..., info: Optional[_InfoType]=..., **kw: Any):
        ...

    @overload
    def __init__(self: 'async_sessionmaker[AsyncSession]', bind: Optional[_AsyncSessionBind]=..., *, autoflush: bool=..., expire_on_commit: bool=..., info: Optional[_InfoType]=..., **kw: Any):
        ...

    def __init__(self, bind: Optional[_AsyncSessionBind]=None, *, class_: Type[_AS]=AsyncSession, autoflush: bool=True, expire_on_commit: bool=True, info: Optional[_InfoType]=None, **kw: Any):
        """Construct a new :class:`.async_sessionmaker`.

        All arguments here except for ``class_`` correspond to arguments
        accepted by :class:`.Session` directly. See the
        :meth:`.AsyncSession.__init__` docstring for more details on
        parameters.


        """
        kw['bind'] = bind
        kw['autoflush'] = autoflush
        kw['expire_on_commit'] = expire_on_commit
        if info is not None:
            kw['info'] = info
        self.kw = kw
        self.class_ = class_

    def begin(self) -> _AsyncSessionContextManager[_AS]:
        """Produce a context manager that both provides a new
        :class:`_orm.AsyncSession` as well as a transaction that commits.


        e.g.::

            async def main():
                Session = async_sessionmaker(some_engine)

                async with Session.begin() as session:
                    session.add(some_object)

                # commits transaction, closes session


        """
        session = self()
        return session._maker_context_manager()

    def __call__(self, **local_kw: Any) -> _AS:
        """Produce a new :class:`.AsyncSession` object using the configuration
        established in this :class:`.async_sessionmaker`.

        In Python, the ``__call__`` method is invoked on an object when
        it is "called" in the same way as a function::

            AsyncSession = async_sessionmaker(async_engine, expire_on_commit=False)
            session = AsyncSession()  # invokes sessionmaker.__call__()

        """
        for k, v in self.kw.items():
            if k == 'info' and 'info' in local_kw:
                d = v.copy()
                d.update(local_kw['info'])
                local_kw['info'] = d
            else:
                local_kw.setdefault(k, v)
        return self.class_(**local_kw)

    def configure(self, **new_kw: Any) -> None:
        """(Re)configure the arguments for this async_sessionmaker.

        e.g.::

            AsyncSession = async_sessionmaker(some_engine)

            AsyncSession.configure(bind=create_async_engine('sqlite+aiosqlite://'))
        """
        self.kw.update(new_kw)

    def __repr__(self) -> str:
        return '%s(class_=%r, %s)' % (self.__class__.__name__, self.class_.__name__, ', '.join(('%s=%r' % (k, v) for k, v in self.kw.items())))