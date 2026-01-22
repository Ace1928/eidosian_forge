from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
class VirtualTimePolicy(enum.Enum):
    """
    advance: If the scheduler runs out of immediate work, the virtual time base may fast forward to
    allow the next delayed task (if any) to run; pause: The virtual time base may not advance;
    pauseIfNetworkFetchesPending: The virtual time base may not advance if there are any pending
    resource fetches.
    """
    ADVANCE = 'advance'
    PAUSE = 'pause'
    PAUSE_IF_NETWORK_FETCHES_PENDING = 'pauseIfNetworkFetchesPending'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)