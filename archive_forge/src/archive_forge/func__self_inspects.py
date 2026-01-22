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
def _self_inspects(cls: _TT) -> _TT:
    if cls in _registrars:
        raise AssertionError('Type %s is already registered' % cls)
    _registrars[cls] = True
    return cls