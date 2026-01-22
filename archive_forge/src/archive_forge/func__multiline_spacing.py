from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
def _multiline_spacing(self, font, spacing, stroke_width):
    return self.textbbox((0, 0), 'A', font, stroke_width=stroke_width)[3] + stroke_width + spacing