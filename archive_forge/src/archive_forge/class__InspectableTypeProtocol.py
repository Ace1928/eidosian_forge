from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Optional
from typing import overload
from typing import Type
from typing import TypeVar
from typing import Union
from . import exc
from .util.typing import Literal
from .util.typing import Protocol
class _InspectableTypeProtocol(Protocol[_TCov]):
    """a protocol defining a method that's used when a type (ie the class
    itself) is passed to inspect().

    """

    def _sa_inspect_type(self) -> _TCov:
        ...