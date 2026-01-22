from __future__ import annotations
import functools
class MinFilter(RankFilter):
    """
    Create a min filter.  Picks the lowest pixel value in a window with the
    given size.

    :param size: The kernel size, in pixels.
    """
    name = 'Min'

    def __init__(self, size=3):
        self.size = size
        self.rank = 0