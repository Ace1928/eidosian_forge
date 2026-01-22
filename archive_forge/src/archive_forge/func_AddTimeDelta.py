from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
def AddTimeDelta(self, delta, calendar=None):
    """Adds a datetime.timdelta to the duration.

    Args:
      delta: A datetime.timedelta object to add.
      calendar: Use duration units larger than hours if True.

    Returns:
      The modified Duration (self).
    """
    if calendar is not None:
        self.calendar = calendar
    self.seconds += delta.total_seconds()
    self._Normalize()
    return self