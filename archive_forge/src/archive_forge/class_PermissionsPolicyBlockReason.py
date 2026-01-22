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
class PermissionsPolicyBlockReason(enum.Enum):
    """
    Reason for a permissions policy feature to be disabled.
    """
    HEADER = 'Header'
    IFRAME_ATTRIBUTE = 'IframeAttribute'
    IN_FENCED_FRAME_TREE = 'InFencedFrameTree'
    IN_ISOLATED_APP = 'InIsolatedApp'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)