from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import service_worker
@event_class('BackgroundService.recordingStateChanged')
@dataclass
class RecordingStateChanged:
    """
    Called when the recording state for the service has been updated.
    """
    is_recording: bool
    service: ServiceName

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RecordingStateChanged:
        return cls(is_recording=bool(json['isRecording']), service=ServiceName.from_json(json['service']))