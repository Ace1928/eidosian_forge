from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import page
class SameSiteCookieExclusionReason(enum.Enum):
    EXCLUDE_SAME_SITE_UNSPECIFIED_TREATED_AS_LAX = 'ExcludeSameSiteUnspecifiedTreatedAsLax'
    EXCLUDE_SAME_SITE_NONE_INSECURE = 'ExcludeSameSiteNoneInsecure'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)