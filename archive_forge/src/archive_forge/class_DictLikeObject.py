from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Protocol, List
import pyglet
from pyglet.font import base
class DictLikeObject(Protocol):

    def get(self, char: str) -> Optional[ImageData]:
        pass