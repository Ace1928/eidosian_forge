from __future__ import annotations
import builtins
import collections.abc as collections_abc
import re
import sys
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import ForwardRef
from typing import Generic
from typing import Iterable
from typing import Mapping
from typing import NewType
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import compat
class DescriptorProto(Protocol):

    def __get__(self, instance: object, owner: Any) -> Any:
        ...

    def __set__(self, instance: Any, value: Any) -> None:
        ...

    def __delete__(self, instance: Any) -> None:
        ...