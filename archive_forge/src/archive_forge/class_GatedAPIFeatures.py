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
class GatedAPIFeatures(enum.Enum):
    SHARED_ARRAY_BUFFERS = 'SharedArrayBuffers'
    SHARED_ARRAY_BUFFERS_TRANSFER_ALLOWED = 'SharedArrayBuffersTransferAllowed'
    PERFORMANCE_MEASURE_MEMORY = 'PerformanceMeasureMemory'
    PERFORMANCE_PROFILE = 'PerformanceProfile'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)