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
def _inspects(*types: Type[Any]) -> Callable[[_F], _F]:

    def decorate(fn_or_cls: _F) -> _F:
        for type_ in types:
            if type_ in _registrars:
                raise AssertionError('Type %s is already registered' % type_)
            _registrars[type_] = fn_or_cls
        return fn_or_cls
    return decorate