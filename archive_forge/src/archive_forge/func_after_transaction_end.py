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
def after_transaction_end(self, session: Session, transaction: SessionTransaction) -> None:
    """Execute when the span of a :class:`.SessionTransaction` ends.

        This event differs from :meth:`~.SessionEvents.after_commit`
        in that it corresponds to all :class:`.SessionTransaction`
        objects in use, including those for nested transactions
        and subtransactions, and is always matched by a corresponding
        :meth:`~.SessionEvents.after_transaction_create` event.

        :param session: the target :class:`.Session`.
        :param transaction: the target :class:`.SessionTransaction`.

         To detect if this is the outermost
         :class:`.SessionTransaction`, as opposed to a "subtransaction" or a
         SAVEPOINT, test that the :attr:`.SessionTransaction.parent` attribute
         is ``None``::

                @event.listens_for(session, "after_transaction_create")
                def after_transaction_end(session, transaction):
                    if transaction.parent is None:
                        # work with top-level transaction

         To detect if the :class:`.SessionTransaction` is a SAVEPOINT, use the
         :attr:`.SessionTransaction.nested` attribute::

                @event.listens_for(session, "after_transaction_create")
                def after_transaction_end(session, transaction):
                    if transaction.nested:
                        # work with SAVEPOINT transaction


        .. seealso::

            :class:`.SessionTransaction`

            :meth:`~.SessionEvents.after_transaction_create`

        """