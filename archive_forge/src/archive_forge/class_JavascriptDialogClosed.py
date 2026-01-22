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
@event_class('Page.javascriptDialogClosed')
@dataclass
class JavascriptDialogClosed:
    """
    Fired when a JavaScript initiated dialog (alert, confirm, prompt, or onbeforeunload) has been
    closed.
    """
    result: bool
    user_input: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> JavascriptDialogClosed:
        return cls(result=bool(json['result']), user_input=str(json['userInput']))