from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
class PreloadingStatus(enum.Enum):
    """
    Preloading status values, see also PreloadingTriggeringOutcome. This
    status is shared by prefetchStatusUpdated and prerenderStatusUpdated.
    """
    PENDING = 'Pending'
    RUNNING = 'Running'
    READY = 'Ready'
    SUCCESS = 'Success'
    FAILURE = 'Failure'
    NOT_SUPPORTED = 'NotSupported'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)