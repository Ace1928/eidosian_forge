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
def after_mapper_constructed(self, mapper: Mapper[_O], class_: Type[_O]) -> None:
    """Receive a class and mapper when the :class:`_orm.Mapper` has been
        fully constructed.

        This event is called after the initial constructor for
        :class:`_orm.Mapper` completes.  This occurs after the
        :meth:`_orm.MapperEvents.instrument_class` event and after the
        :class:`_orm.Mapper` has done an initial pass of its arguments
        to generate its collection of :class:`_orm.MapperProperty` objects,
        which are accessible via the :meth:`_orm.Mapper.get_property`
        method and the :attr:`_orm.Mapper.iterate_properties` attribute.

        This event differs from the
        :meth:`_orm.MapperEvents.before_mapper_configured` event in that it
        is invoked within the constructor for :class:`_orm.Mapper`, rather
        than within the :meth:`_orm.registry.configure` process.   Currently,
        this event is the only one which is appropriate for handlers that
        wish to create additional mapped classes in response to the
        construction of this :class:`_orm.Mapper`, which will be part of the
        same configure step when :meth:`_orm.registry.configure` next runs.

        .. versionadded:: 2.0.2

        .. seealso::

            :ref:`examples_versioning` - an example which illustrates the use
            of the :meth:`_orm.MapperEvents.before_mapper_configured`
            event to create new mappers to record change-audit histories on
            objects.

        """