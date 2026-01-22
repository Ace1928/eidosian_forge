from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import instrumentation
from . import interfaces
from . import mapperlib
from .attributes import QueryableAttribute
from .base import _mapper_or_none
from .base import NO_KEY
from .instrumentation import ClassManager
from .instrumentation import InstrumentationFactory
from .query import BulkDelete
from .query import BulkUpdate
from .query import Query
from .scoping import scoped_session
from .session import Session
from .session import sessionmaker
from .. import event
from .. import exc
from .. import util
from ..event import EventTarget
from ..event.registry import _ET
from ..util.compat import inspect_getfullargspec
def after_delete(self, mapper: Mapper[_O], connection: Connection, target: _O) -> None:
    """Receive an object instance after a DELETE statement
        has been emitted corresponding to that instance.

        .. note:: this event **only** applies to the
           :ref:`session flush operation <session_flushing>`
           and does **not** apply to the ORM DML operations described at
           :ref:`orm_expression_update_delete`.  To intercept ORM
           DML events, use :meth:`_orm.SessionEvents.do_orm_execute`.

        This event is used to emit additional SQL statements on
        the given connection as well as to perform application
        specific bookkeeping related to a deletion event.

        The event is often called for a batch of objects of the
        same class after their DELETE statements have been emitted at
        once in a previous step.

        .. warning::

            Mapper-level flush events only allow **very limited operations**,
            on attributes local to the row being operated upon only,
            as well as allowing any SQL to be emitted on the given
            :class:`_engine.Connection`.  **Please read fully** the notes
            at :ref:`session_persistence_mapper` for guidelines on using
            these methods; generally, the :meth:`.SessionEvents.before_flush`
            method should be preferred for general on-flush changes.

        :param mapper: the :class:`_orm.Mapper` which is the target
         of this event.
        :param connection: the :class:`_engine.Connection` being used to
         emit DELETE statements for this instance.  This
         provides a handle into the current transaction on the
         target database specific to this instance.
        :param target: the mapped instance being deleted.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :return: No return value is supported by this event.

        .. seealso::

            :ref:`session_persistence_events`

        """