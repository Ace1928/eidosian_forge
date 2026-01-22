from __future__ import division
import datetime
import math
class AnimatedMarker(Widget):
    """An animated marker for the progress bar which defaults to appear as if
    it were rotating.
    """
    __slots__ = ('markers', 'curmark')

    def __init__(self, markers='|/-\\'):
        self.markers = markers
        self.curmark = -1

    def update(self, pbar):
        """Updates the widget to show the next marker or the first marker when
        finished"""
        if pbar.finished:
            return self.markers[0]
        self.curmark = (self.curmark + 1) % len(self.markers)
        return self.markers[self.curmark]