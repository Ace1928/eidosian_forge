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
@event_class('Page.screencastFrame')
@dataclass
class ScreencastFrame:
    """
    **EXPERIMENTAL**

    Compressed image data requested by the ``startScreencast``.
    """
    data: str
    metadata: ScreencastFrameMetadata
    session_id: int

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ScreencastFrame:
        return cls(data=str(json['data']), metadata=ScreencastFrameMetadata.from_json(json['metadata']), session_id=int(json['sessionId']))