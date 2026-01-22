from __future__ import division
import datetime
import math
class SimpleProgress(Widget):
    """Returns progress as a count of the total (e.g.: "5 of 47")."""
    __slots__ = ('sep',)

    def __init__(self, sep=' of '):
        self.sep = sep

    def update(self, pbar):
        return '%d%s%s' % (pbar.currval, self.sep, pbar.maxval)