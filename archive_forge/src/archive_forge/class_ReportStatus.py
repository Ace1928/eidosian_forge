from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
class ReportStatus(enum.Enum):
    """
    The status of a Reporting API report.
    """
    QUEUED = 'Queued'
    PENDING = 'Pending'
    MARKED_FOR_REMOVAL = 'MarkedForRemoval'
    SUCCESS = 'Success'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)