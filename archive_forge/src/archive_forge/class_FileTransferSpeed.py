from __future__ import division
import datetime
import math
class FileTransferSpeed(Widget):
    """Widget for showing the transfer speed (useful for file transfers)."""
    FMT = '%6.2f %s%s/s'
    PREFIXES = ' kMGTPEZY'
    __slots__ = ('unit',)

    def __init__(self, unit='B'):
        self.unit = unit

    def update(self, pbar):
        """Updates the widget with the current SI prefixed speed."""
        if pbar.seconds_elapsed < 2e-06 or pbar.currval < 2e-06:
            scaled = power = 0
        else:
            speed = pbar.currval / pbar.seconds_elapsed
            power = int(math.log(speed, 1000))
            scaled = speed / 1000.0 ** power
        return self.FMT % (scaled, self.PREFIXES[power], self.unit)