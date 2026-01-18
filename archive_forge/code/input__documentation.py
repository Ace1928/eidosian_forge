from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

    Synthesizes a tap gesture over a time period by issuing appropriate touch events.

    **EXPERIMENTAL**

    :param x: X coordinate of the start of the gesture in CSS pixels.
    :param y: Y coordinate of the start of the gesture in CSS pixels.
    :param duration: *(Optional)* Duration between touchdown and touchup events in ms (default: 50).
    :param tap_count: *(Optional)* Number of times to perform the tap (e.g. 2 for double tap, default: 1).
    :param gesture_source_type: *(Optional)* Which type of input events to be generated (default: 'default', which queries the platform for the preferred input type).
    