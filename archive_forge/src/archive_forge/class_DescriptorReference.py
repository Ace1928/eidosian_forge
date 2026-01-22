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
class DescriptorReference(Generic[_DESC]):
    """a descriptor that refers to a descriptor.

    used for cases where we need to have an instance variable referring to an
    object that is itself a descriptor, which typically confuses typing tools
    as they don't know when they should use ``__get__`` or not when referring
    to the descriptor assignment as an instance variable. See
    sqlalchemy.orm.interfaces.PropComparator.prop

    """
    if TYPE_CHECKING:

        def __get__(self, instance: object, owner: Any) -> _DESC:
            ...

        def __set__(self, instance: Any, value: _DESC) -> None:
            ...

        def __delete__(self, instance: Any) -> None:
            ...