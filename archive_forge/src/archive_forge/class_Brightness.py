from __future__ import annotations
from . import Image, ImageFilter, ImageStat
class Brightness(_Enhance):
    """Adjust image brightness.

    This class can be used to control the brightness of an image.  An
    enhancement factor of 0.0 gives a black image. A factor of 1.0 gives the
    original image.
    """

    def __init__(self, image):
        self.image = image
        self.degenerate = Image.new(image.mode, image.size, 0)
        if 'A' in image.getbands():
            self.degenerate.putalpha(image.getchannel('A'))