import numpy as np
from ..core import Format
class ImageExtension(bsdf.Extension):
    """We implement two extensions that trigger on the Image classes."""

    def encode(self, s, v):
        return dict(array=v.array, meta=v.meta)

    def decode(self, s, v):
        return Image(v['array'], v['meta'])