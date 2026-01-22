from __future__ import annotations
import functools
import operator
import re
from . import ExifTags, Image, ImagePalette
def _lut(image, lut):
    if image.mode == 'P':
        msg = 'mode P support coming soon'
        raise NotImplementedError(msg)
    elif image.mode in ('L', 'RGB'):
        if image.mode == 'RGB' and len(lut) == 256:
            lut = lut + lut + lut
        return image.point(lut)
    else:
        msg = f'not supported for mode {image.mode}'
        raise OSError(msg)