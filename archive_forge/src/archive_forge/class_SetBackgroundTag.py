import os
import zlib
import time  # noqa
import logging
import numpy as np
class SetBackgroundTag(ControlTag):
    """Set the color in 0-255, or 0-1 (if floats given)."""

    def __init__(self, *rgb):
        self.tagtype = 9
        if len(rgb) == 1:
            rgb = rgb[0]
        self.rgb = rgb

    def process_tag(self):
        bb = bytes()
        for i in range(3):
            clr = self.rgb[i]
            if isinstance(clr, float):
                clr = clr * 255
            bb += int2uint8(clr)
        self.bytes = bb