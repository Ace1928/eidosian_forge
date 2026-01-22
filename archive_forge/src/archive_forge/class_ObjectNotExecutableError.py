from __future__ import annotations
import typing
from typing import Any
from typing import List
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import Union
from .util import compat
from .util import preloaded as _preloaded
class ObjectNotExecutableError(ArgumentError):
    """Raised when an object is passed to .execute() that can't be
    executed as SQL.

    """

    def __init__(self, target: Any):
        super().__init__('Not an executable object: %r' % target)
        self.target = target

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return (self.__class__, (self.target,))