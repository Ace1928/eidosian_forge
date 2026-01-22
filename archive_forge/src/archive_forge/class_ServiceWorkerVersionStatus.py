from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import target
class ServiceWorkerVersionStatus(enum.Enum):
    NEW = 'new'
    INSTALLING = 'installing'
    INSTALLED = 'installed'
    ACTIVATING = 'activating'
    ACTIVATED = 'activated'
    REDUNDANT = 'redundant'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)