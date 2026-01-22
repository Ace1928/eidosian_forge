import datetime
import numbers
import abc
import bisect
import pytz
class PeriodicCommand(DelayedCommand):
    """
    Like a delayed command, but expect this command to run every delay
    seconds.
    """

    def _next_time(self):
        """
        Add delay to self, localized
        """
        return self._localize(self + self.delay)

    @staticmethod
    def _localize(dt):
        """
        Rely on pytz.localize to ensure new result honors DST.
        """
        try:
            tz = dt.tzinfo
            return tz.localize(dt.replace(tzinfo=None))
        except AttributeError:
            return dt

    def next(self):
        cmd = self.__class__.from_datetime(self._next_time())
        cmd.delay = self.delay
        cmd.target = self.target
        return cmd

    def __setattr__(self, key, value):
        if key == 'delay' and (not value > datetime.timedelta()):
            raise ValueError('A PeriodicCommand must have a positive, non-zero delay.')
        super().__setattr__(key, value)