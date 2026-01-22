from __future__ import annotations
import logging # isort:skip
from typing import ClassVar, Iterator
from .color import RGB
class ColorGroup(metaclass=_ColorGroupMeta):
    """ Collect a group of named colors into an iterable, indexable group.

    """