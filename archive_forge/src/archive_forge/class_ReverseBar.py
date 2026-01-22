from __future__ import division
import datetime
import math
class ReverseBar(Bar):
    """A bar which has a marker which bounces from side to side."""

    def __init__(self, marker='#', left='|', right='|', fill=' ', fill_left=False):
        """Creates a customizable progress bar.

        marker - string or updatable object to use as a marker
        left - string or updatable object to use as a left border
        right - string or updatable object to use as a right border
        fill - character to use for the empty part of the progress bar
        fill_left - whether to fill from the left or the right
        """
        self.marker = marker
        self.left = left
        self.right = right
        self.fill = fill
        self.fill_left = fill_left