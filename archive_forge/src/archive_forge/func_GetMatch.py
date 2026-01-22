from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import calendar
import datetime
from . import groc
def GetMatch(self, start):
    """Returns the next match after time start.

    Must be implemented in subclasses.

    Arguments:
      start: a datetime to start from. Matches will start from after this time.
        This may be in any pytz time zone, or it may be timezone-naive
        (interpreted as UTC).

    Returns:
      a datetime object in the timezone of the input 'start'
    """
    start_time = _ToTimeZone(start, self.timezone).replace(tzinfo=None)
    if self.months:
        months = self._NextMonthGenerator(start_time.month, self.months)
    while True:
        month, yearwraps = next(months)
        candidate_month = start_time.replace(day=1, month=month, year=start_time.year + yearwraps)
        day_matches = self._MatchingDays(candidate_month.year, month)
        if (candidate_month.year, candidate_month.month) == (start_time.year, start_time.month):
            day_matches = [x for x in day_matches if x >= start_time.day]
        while day_matches:
            out = candidate_month.replace(day=day_matches[0], hour=self.time.hour, minute=self.time.minute, second=0, microsecond=0)
            if self.timezone and pytz is not None:
                try:
                    out = self.timezone.localize(out, is_dst=None)
                except AmbiguousTimeError:
                    start_utc = _ToTimeZone(start, pytz.utc)
                    dst_doubled_time_first_match_utc = _ToTimeZone(self.timezone.localize(out, is_dst=True), pytz.utc)
                    if start_utc < dst_doubled_time_first_match_utc:
                        out = self.timezone.localize(out, is_dst=True)
                    else:
                        out = self.timezone.localize(out, is_dst=False)
                except NonExistentTimeError:
                    day_matches.pop(0)
                    continue
            if start < _ToTimeZone(out, start.tzinfo):
                return _ToTimeZone(out, start.tzinfo)
            else:
                day_matches.pop(0)