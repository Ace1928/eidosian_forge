from __future__ import annotations
import logging # isort:skip
from ..core.enums import TextureRepetition
from ..core.has_props import abstract
from ..core.properties import Enum, Required, String
from ..model import Model
class CanvasTexture(Texture):
    """

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    code = Required(String, help='\n    A snippet of JavaScript code to execute in the browser.\n\n    ')