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
def after_configured(self) -> None:
    """Called after a series of mappers have been configured.

        The :meth:`.MapperEvents.after_configured` event is invoked
        each time the :func:`_orm.configure_mappers` function is
        invoked, after the function has completed its work.
        :func:`_orm.configure_mappers` is typically invoked
        automatically as mappings are first used, as well as each time
        new mappers have been made available and new mapper use is
        detected.

        Contrast this event to the :meth:`.MapperEvents.mapper_configured`
        event, which is called on a per-mapper basis while the configuration
        operation proceeds; unlike that event, when this event is invoked,
        all cross-configurations (e.g. backrefs) will also have been made
        available for any mappers that were pending.
        Also contrast to :meth:`.MapperEvents.before_configured`,
        which is invoked before the series of mappers has been configured.

        This event can **only** be applied to the :class:`_orm.Mapper` class,
        and not to individual mappings or
        mapped classes.  It is only invoked for all mappings as a whole::

            from sqlalchemy.orm import Mapper

            @event.listens_for(Mapper, "after_configured")
            def go():
                # ...

        Theoretically this event is called once per
        application, but is actually called any time new mappers
        have been affected by a :func:`_orm.configure_mappers`
        call.   If new mappings are constructed after existing ones have
        already been used, this event will likely be called again.  To ensure
        that a particular event is only called once and no further, the
        ``once=True`` argument (new in 0.9.4) can be applied::

            from sqlalchemy.orm import mapper

            @event.listens_for(mapper, "after_configured", once=True)
            def go():
                # ...

        .. seealso::

            :meth:`.MapperEvents.before_mapper_configured`

            :meth:`.MapperEvents.mapper_configured`

            :meth:`.MapperEvents.before_configured`

        """