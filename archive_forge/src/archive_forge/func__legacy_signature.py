from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from .registry import _ET
from .registry import _ListenerFnType
from .. import util
from ..util.compat import FullArgSpec
def _legacy_signature(since: str, argnames: List[str], converter: Optional[Callable[..., Any]]=None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """legacy sig decorator


    :param since: string version for deprecation warning
    :param argnames: list of strings, which is *all* arguments that the legacy
     version accepted, including arguments that are still there
    :param converter: lambda that will accept tuple of this full arg signature
     and return tuple of new arg signature.

    """

    def leg(fn: Callable[..., Any]) -> Callable[..., Any]:
        if not hasattr(fn, '_legacy_signatures'):
            fn._legacy_signatures = []
        fn._legacy_signatures.append((since, argnames, converter))
        return fn
    return leg