from __future__ import annotations
import functools
class BoxBlur(MultibandFilter):
    """Blurs the image by setting each pixel to the average value of the pixels
    in a square box extending radius pixels in each direction.
    Supports float radius of arbitrary size. Uses an optimized implementation
    which runs in linear time relative to the size of the image
    for any radius value.

    :param radius: Size of the box in a direction. Either a sequence of two numbers for
                   x and y, or a single number for both.

                   Radius 0 does not blur, returns an identical image.
                   Radius 1 takes 1 pixel in each direction, i.e. 9 pixels in total.
    """
    name = 'BoxBlur'

    def __init__(self, radius):
        xy = radius
        if not isinstance(xy, (tuple, list)):
            xy = (xy, xy)
        if xy[0] < 0 or xy[1] < 0:
            msg = 'radius must be >= 0'
            raise ValueError(msg)
        self.radius = radius

    def filter(self, image):
        xy = self.radius
        if not isinstance(xy, (tuple, list)):
            xy = (xy, xy)
        if xy == (0, 0):
            return image.copy()
        return image.box_blur(xy)