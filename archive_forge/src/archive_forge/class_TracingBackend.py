from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
class TracingBackend(enum.Enum):
    """
    Backend type to use for tracing. ``chrome`` uses the Chrome-integrated
    tracing service and is supported on all platforms. ``system`` is only
    supported on Chrome OS and uses the Perfetto system tracing service.
    ``auto`` chooses ``system`` when the perfettoConfig provided to Tracing.start
    specifies at least one non-Chrome data source; otherwise uses ``chrome``.
    """
    AUTO = 'auto'
    CHROME = 'chrome'
    SYSTEM = 'system'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)