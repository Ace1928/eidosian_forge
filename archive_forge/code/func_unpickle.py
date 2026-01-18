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
def unpickle(self, target: _O, state_dict: _InstanceDict) -> None:
    """Receive an object instance after its associated state has
        been unpickled.

        :param target: the mapped instance.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :param state_dict: the dictionary sent to
         :class:`.InstanceState.__setstate__`, containing the state
         dictionary which was pickled.

        """