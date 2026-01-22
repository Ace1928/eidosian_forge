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
class _InspectableProtocol(Protocol[_TCov]):
    """a protocol defining a method that's used when an instance is
    passed to inspect().

    """

    def _sa_inspect_instance(self) -> _TCov:
        ...