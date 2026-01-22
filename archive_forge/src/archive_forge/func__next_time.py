import datetime
import numbers
import abc
import bisect
import pytz
def _next_time(self):
    """
        Add delay to self, localized
        """
    return self._localize(self + self.delay)