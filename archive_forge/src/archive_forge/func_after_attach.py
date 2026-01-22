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
@_lifecycle_event
def after_attach(self, session: Session, instance: _O) -> None:
    """Execute after an instance is attached to a session.

        This is called after an add, delete or merge.

        .. note::

           As of 0.8, this event fires off *after* the item
           has been fully associated with the session, which is
           different than previous releases.  For event
           handlers that require the object not yet
           be part of session state (such as handlers which
           may autoflush while the target object is not
           yet complete) consider the
           new :meth:`.before_attach` event.

        .. seealso::

            :meth:`~.SessionEvents.before_attach`

            :ref:`session_lifecycle_events`

        """