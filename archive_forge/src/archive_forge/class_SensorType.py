from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
class SensorType(enum.Enum):
    """
    Used to specify sensor types to emulate.
    See https://w3c.github.io/sensors/#automation for more information.
    """
    ABSOLUTE_ORIENTATION = 'absolute-orientation'
    ACCELEROMETER = 'accelerometer'
    AMBIENT_LIGHT = 'ambient-light'
    GRAVITY = 'gravity'
    GYROSCOPE = 'gyroscope'
    LINEAR_ACCELERATION = 'linear-acceleration'
    MAGNETOMETER = 'magnetometer'
    PROXIMITY = 'proximity'
    RELATIVE_ORIENTATION = 'relative-orientation'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)