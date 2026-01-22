from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .session import _S
from .session import Session
from .. import exc as sa_exc
from .. import util
from ..util import create_proxy_methods
from ..util import ScopedRegistry
from ..util import ThreadLocalRegistry
from ..util import warn
from ..util import warn_deprecated
from ..util.typing import Protocol
class query:

    def __get__(s, instance: Any, owner: Type[_O]) -> Query[_O]:
        if query_cls:
            return query_cls(owner, session=self.registry())
        else:
            return self.registry().query(owner)