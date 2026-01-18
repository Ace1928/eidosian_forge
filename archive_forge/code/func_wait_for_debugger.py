from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def wait_for_debugger() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Pauses page execution. Can be resumed using generic Runtime.runIfWaitingForDebugger.

    **EXPERIMENTAL**
    """
    cmd_dict: T_JSON_DICT = {'method': 'Page.waitForDebugger'}
    json = (yield cmd_dict)