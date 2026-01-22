from __future__ import annotations
import logging # isort:skip
from ..core.enums import TextureRepetition
from ..core.has_props import abstract
from ..core.properties import Enum, Required, String
from ..model import Model
class ImageURLTexture(Texture):
    """

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    url = Required(String, help='\n    A URL to a drawable resource like image, video, etc.\n\n    ')