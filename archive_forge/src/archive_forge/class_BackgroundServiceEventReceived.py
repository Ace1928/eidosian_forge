from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import service_worker
@event_class('BackgroundService.backgroundServiceEventReceived')
@dataclass
class BackgroundServiceEventReceived:
    """
    Called with all existing backgroundServiceEvents when enabled, and all new
    events afterwards if enabled and recording.
    """
    background_service_event: BackgroundServiceEvent

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> BackgroundServiceEventReceived:
        return cls(background_service_event=BackgroundServiceEvent.from_json(json['backgroundServiceEvent']))