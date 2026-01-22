from __future__ import annotations
import functools
class MedianFilter(RankFilter):
    """
    Create a median filter. Picks the median pixel value in a window with the
    given size.

    :param size: The kernel size, in pixels.
    """
    name = 'Median'

    def __init__(self, size=3):
        self.size = size
        self.rank = size * size // 2