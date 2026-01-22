from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import calendar
import datetime
from . import groc
def _TimeIsInRange(self, t):
    """Returns true if 't' falls between start_time and end_time, inclusive.

    Arguments:
      t: a datetime object, in self.timezone

    Returns:
      a boolean
    """
    previous_start_time = self._GetPreviousDateTime(t, self.start_time, self.timezone)
    previous_end_time = self._GetPreviousDateTime(t, self.end_time, self.timezone)
    if previous_start_time > previous_end_time:
        return True
    else:
        return t == previous_end_time