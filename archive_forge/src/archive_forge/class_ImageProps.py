from __future__ import annotations
import logging # isort:skip
from .enums import (
from .has_props import HasProps
from .properties import (
class ImageProps(HasProps):
    """ Properties relevant to rendering images.

    Mirrors the BokehJS ``properties.ImageVector`` class.

    """
    global_alpha = AlphaSpec(help=_alpha_help % 'images')