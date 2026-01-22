from __future__ import annotations
import contextlib
from enum import Enum
import itertools
import sys
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import bulk_persistence
from . import context
from . import descriptor_props
from . import exc
from . import identity
from . import loading
from . import query
from . import state as statelib
from ._typing import _O
from ._typing import insp_is_mapper
from ._typing import is_composite_class
from ._typing import is_orm_option
from ._typing import is_user_defined_option
from .base import _class_to_mapper
from .base import _none_set
from .base import _state_mapper
from .base import instance_str
from .base import LoaderCallableStatus
from .base import object_mapper
from .base import object_state
from .base import PassiveFlag
from .base import state_str
from .context import FromStatement
from .context import ORMCompileState
from .identity import IdentityMap
from .query import Query
from .state import InstanceState
from .state_changes import _StateChange
from .state_changes import _StateChangeState
from .state_changes import _StateChangeStates
from .unitofwork import UOWTransaction
from .. import engine
from .. import exc as sa_exc
from .. import sql
from .. import util
from ..engine import Connection
from ..engine import Engine
from ..engine.util import TransactionalContext
from ..event import dispatcher
from ..event import EventTarget
from ..inspection import inspect
from ..inspection import Inspectable
from ..sql import coercions
from ..sql import dml
from ..sql import roles
from ..sql import Select
from ..sql import TableClause
from ..sql import visitors
from ..sql.base import _NoArg
from ..sql.base import CompileState
from ..sql.schema import Table
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import IdentitySet
from ..util.typing import Literal
from ..util.typing import Protocol
class sessionmaker(_SessionClassMethods, Generic[_S]):
    """A configurable :class:`.Session` factory.

    The :class:`.sessionmaker` factory generates new
    :class:`.Session` objects when called, creating them given
    the configurational arguments established here.

    e.g.::

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        # an Engine, which the Session will use for connection
        # resources
        engine = create_engine('postgresql+psycopg2://scott:tiger@localhost/')

        Session = sessionmaker(engine)

        with Session() as session:
            session.add(some_object)
            session.add(some_other_object)
            session.commit()

    Context manager use is optional; otherwise, the returned
    :class:`_orm.Session` object may be closed explicitly via the
    :meth:`_orm.Session.close` method.   Using a
    ``try:/finally:`` block is optional, however will ensure that the close
    takes place even if there are database errors::

        session = Session()
        try:
            session.add(some_object)
            session.add(some_other_object)
            session.commit()
        finally:
            session.close()

    :class:`.sessionmaker` acts as a factory for :class:`_orm.Session`
    objects in the same way as an :class:`_engine.Engine` acts as a factory
    for :class:`_engine.Connection` objects.  In this way it also includes
    a :meth:`_orm.sessionmaker.begin` method, that provides a context
    manager which both begins and commits a transaction, as well as closes
    out the :class:`_orm.Session` when complete, rolling back the transaction
    if any errors occur::

        Session = sessionmaker(engine)

        with Session.begin() as session:
            session.add(some_object)
            session.add(some_other_object)
        # commits transaction, closes session

    .. versionadded:: 1.4

    When calling upon :class:`_orm.sessionmaker` to construct a
    :class:`_orm.Session`, keyword arguments may also be passed to the
    method; these arguments will override that of the globally configured
    parameters.  Below we use a :class:`_orm.sessionmaker` bound to a certain
    :class:`_engine.Engine` to produce a :class:`_orm.Session` that is instead
    bound to a specific :class:`_engine.Connection` procured from that engine::

        Session = sessionmaker(engine)

        # bind an individual session to a connection

        with engine.connect() as connection:
            with Session(bind=connection) as session:
                # work with session

    The class also includes a method :meth:`_orm.sessionmaker.configure`, which
    can be used to specify additional keyword arguments to the factory, which
    will take effect for subsequent :class:`.Session` objects generated. This
    is usually used to associate one or more :class:`_engine.Engine` objects
    with an existing
    :class:`.sessionmaker` factory before it is first used::

        # application starts, sessionmaker does not have
        # an engine bound yet
        Session = sessionmaker()

        # ... later, when an engine URL is read from a configuration
        # file or other events allow the engine to be created
        engine = create_engine('sqlite:///foo.db')
        Session.configure(bind=engine)

        sess = Session()
        # work with session

    .. seealso::

        :ref:`session_getting` - introductory text on creating
        sessions using :class:`.sessionmaker`.

    """
    class_: Type[_S]

    @overload
    def __init__(self, bind: Optional[_SessionBind]=..., *, class_: Type[_S], autoflush: bool=..., expire_on_commit: bool=..., info: Optional[_InfoType]=..., **kw: Any):
        ...

    @overload
    def __init__(self: 'sessionmaker[Session]', bind: Optional[_SessionBind]=..., *, autoflush: bool=..., expire_on_commit: bool=..., info: Optional[_InfoType]=..., **kw: Any):
        ...

    def __init__(self, bind: Optional[_SessionBind]=None, *, class_: Type[_S]=Session, autoflush: bool=True, expire_on_commit: bool=True, info: Optional[_InfoType]=None, **kw: Any):
        """Construct a new :class:`.sessionmaker`.

        All arguments here except for ``class_`` correspond to arguments
        accepted by :class:`.Session` directly.  See the
        :meth:`.Session.__init__` docstring for more details on parameters.

        :param bind: a :class:`_engine.Engine` or other :class:`.Connectable`
         with
         which newly created :class:`.Session` objects will be associated.
        :param class\\_: class to use in order to create new :class:`.Session`
         objects.  Defaults to :class:`.Session`.
        :param autoflush: The autoflush setting to use with newly created
         :class:`.Session` objects.

         .. seealso::

            :ref:`session_flushing` - additional background on autoflush

        :param expire_on_commit=True: the
         :paramref:`_orm.Session.expire_on_commit` setting to use
         with newly created :class:`.Session` objects.

        :param info: optional dictionary of information that will be available
         via :attr:`.Session.info`.  Note this dictionary is *updated*, not
         replaced, when the ``info`` parameter is specified to the specific
         :class:`.Session` construction operation.

        :param \\**kw: all other keyword arguments are passed to the
         constructor of newly created :class:`.Session` objects.

        """
        kw['bind'] = bind
        kw['autoflush'] = autoflush
        kw['expire_on_commit'] = expire_on_commit
        if info is not None:
            kw['info'] = info
        self.kw = kw
        self.class_ = type(class_.__name__, (class_,), {})

    def begin(self) -> contextlib.AbstractContextManager[_S]:
        """Produce a context manager that both provides a new
        :class:`_orm.Session` as well as a transaction that commits.


        e.g.::

            Session = sessionmaker(some_engine)

            with Session.begin() as session:
                session.add(some_object)

            # commits transaction, closes session

        .. versionadded:: 1.4


        """
        session = self()
        return session._maker_context_manager()

    def __call__(self, **local_kw: Any) -> _S:
        """Produce a new :class:`.Session` object using the configuration
        established in this :class:`.sessionmaker`.

        In Python, the ``__call__`` method is invoked on an object when
        it is "called" in the same way as a function::

            Session = sessionmaker(some_engine)
            session = Session()  # invokes sessionmaker.__call__()

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
        """(Re)configure the arguments for this sessionmaker.

        e.g.::

            Session = sessionmaker()

            Session.configure(bind=create_engine('sqlite://'))
        """
        self.kw.update(new_kw)

    def __repr__(self) -> str:
        return '%s(class_=%r, %s)' % (self.__class__.__name__, self.class_.__name__, ', '.join(('%s=%r' % (k, v) for k, v in self.kw.items())))