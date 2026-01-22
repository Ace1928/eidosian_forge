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
class OriginTrialStatus(enum.Enum):
    """
    Status for an Origin Trial.
    """
    ENABLED = 'Enabled'
    VALID_TOKEN_NOT_PROVIDED = 'ValidTokenNotProvided'
    OS_NOT_SUPPORTED = 'OSNotSupported'
    TRIAL_NOT_ALLOWED = 'TrialNotAllowed'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)