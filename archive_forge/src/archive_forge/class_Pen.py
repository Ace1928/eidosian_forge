from __future__ import annotations
from . import Image, ImageColor, ImageDraw, ImageFont, ImagePath
class Pen:
    """Stores an outline color and width."""

    def __init__(self, color, width=1, opacity=255):
        self.color = ImageColor.getrgb(color)
        self.width = width