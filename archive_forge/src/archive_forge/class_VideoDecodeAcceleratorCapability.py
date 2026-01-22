from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class VideoDecodeAcceleratorCapability:
    """
    Describes a supported video decoding profile with its associated minimum and
    maximum resolutions.
    """
    profile: str
    max_resolution: Size
    min_resolution: Size

    def to_json(self):
        json = dict()
        json['profile'] = self.profile
        json['maxResolution'] = self.max_resolution.to_json()
        json['minResolution'] = self.min_resolution.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(profile=str(json['profile']), max_resolution=Size.from_json(json['maxResolution']), min_resolution=Size.from_json(json['minResolution']))