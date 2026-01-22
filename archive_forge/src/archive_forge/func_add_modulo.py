from __future__ import annotations
from . import Image
def add_modulo(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Add two images, without clipping the result. ::

        out = ((image1 + image2) % MAX)

    :rtype: :py:class:`~PIL.Image.Image`
    """
    image1.load()
    image2.load()
    return image1._new(image1.im.chop_add_modulo(image2.im))