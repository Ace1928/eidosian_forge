from __future__ import annotations
from inspect import iscoroutine
from io import BytesIO
from sys import exc_info
from traceback import extract_tb
from types import GeneratorType
from typing import (
from twisted.internet.defer import Deferred, ensureDeferred
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.web._stan import CDATA, CharRef, Comment, Tag, slot, voidElements
from twisted.web.error import FlattenerError, UnfilledSlot, UnsupportedType
from twisted.web.iweb import IRenderable, IRequest
def _getSlotValue(name: str, slotData: Sequence[Optional[Mapping[str, Flattenable]]], default: Optional[Flattenable]=None) -> Flattenable:
    """
    Find the value of the named slot in the given stack of slot data.
    """
    for slotFrame in reversed(slotData):
        if slotFrame is not None and name in slotFrame:
            return slotFrame[name]
    else:
        if default is not None:
            return default
        raise UnfilledSlot(name)