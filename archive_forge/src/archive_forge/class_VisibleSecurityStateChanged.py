from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
@event_class('Security.visibleSecurityStateChanged')
@dataclass
class VisibleSecurityStateChanged:
    """
    **EXPERIMENTAL**

    The security state of the page changed.
    """
    visible_security_state: VisibleSecurityState

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> VisibleSecurityStateChanged:
        return cls(visible_security_state=VisibleSecurityState.from_json(json['visibleSecurityState']))