from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class ImageType(enum.Enum):
    """
    Image format of a given image.
    """
    JPEG = 'jpeg'
    WEBP = 'webp'
    UNKNOWN = 'unknown'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)