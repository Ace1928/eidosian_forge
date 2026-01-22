from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import calendar
import datetime
from . import groc
class SpecificTimeSpecification(TimeSpecification):
    """Specific time specification.

  A Specific interval is more complex, but defines a certain time to run and
  the days that it should run. It has the following attributes:
  time     - the time of day to run, as 'HH:MM'
  ordinals - first, second, third &c, as a set of integers in 1..5
  months   - the months that this should run, as a set of integers in 1..12
  weekdays - the days of the week that this should run, as a set of integers,
             0=Sunday, 6=Saturday
  timezone - the optional timezone as a string for this specification.
             Defaults to UTC - valid entries are things like Australia/Victoria
             or PST8PDT.

  A specific time schedule can be quite complex. A schedule could look like
  this:
  '1st,third sat,sun of jan,feb,mar 09:15'

  In this case, ordinals would be {1,3}, weekdays {0,6}, months {1,2,3} and
  time would be '09:15'.
  """

    def __init__(self, ordinals=None, weekdays=None, months=None, monthdays=None, timestr='00:00', timezone=None):
        super(SpecificTimeSpecification, self).__init__()
        if weekdays and monthdays:
            raise ValueError('cannot supply both monthdays and weekdays')
        if ordinals is None:
            self.ordinals = set(range(1, 6))
        else:
            self.ordinals = set(ordinals)
            if self.ordinals and (min(self.ordinals) < 1 or max(self.ordinals) > 5):
                raise ValueError('ordinals must be between 1 and 5 inclusive, got %r' % ordinals)
        if weekdays is None:
            self.weekdays = set(range(7))
        else:
            self.weekdays = set(weekdays)
            if self.weekdays and (min(self.weekdays) < 0 or max(self.weekdays) > 6):
                raise ValueError('weekdays must be between 0 (sun) and 6 (sat) inclusive, got %r' % weekdays)
        if months is None:
            self.months = set(range(1, 13))
        else:
            self.months = set(months)
            if self.months and (min(self.months) < 1 or max(self.months) > 12):
                raise ValueError('months must be between 1 (jan) and 12 (dec) inclusive, got %r' % months)
        if not monthdays:
            self.monthdays = set()
        else:
            if min(monthdays) < 1:
                raise ValueError('day of month must be greater than 0')
            if max(monthdays) > 31:
                raise ValueError('day of month must be less than 32')
            if self.months:
                for month in self.months:
                    _, ndays = calendar.monthrange(4, month)
                    if min(monthdays) <= ndays:
                        break
                else:
                    raise ValueError('invalid day of month, got day %r of month %r' % (max(monthdays), month))
            self.monthdays = set(monthdays)
        self.time = _GetTime(timestr)
        self.timezone = _GetTimezone(timezone)

    def _MatchingDays(self, year, month):
        """Returns matching days for the given year and month.

    For the given year and month, return the days that match this instance's
    day specification, based on either (a) the ordinals and weekdays, or
    (b) the explicitly specified monthdays.  If monthdays are specified,
    dates that fall outside the range of the month will not be returned.

    Arguments:
      year: the year as an integer
      month: the month as an integer, in range 1-12

    Returns:
      a list of matching days, as ints in range 1-31
    """
        start_day, last_day = calendar.monthrange(year, month)
        if self.monthdays:
            return sorted([day for day in self.monthdays if day <= last_day])
        out_days = []
        start_day = (start_day + 1) % 7
        for ordinal in self.ordinals:
            for weekday in self.weekdays:
                day = (weekday - start_day) % 7 + 1
                day += 7 * (ordinal - 1)
                if day <= last_day:
                    out_days.append(day)
        return sorted(out_days)

    def _NextMonthGenerator(self, start, matches):
        """Creates a generator that produces results from the set 'matches'.

    Matches must be >= 'start'. If none match, the wrap counter is incremented,
    and the result set is reset to the full set. Yields a 2-tuple of (match,
    wrapcount).

    Arguments:
      start: first set of matches will be >= this value (an int)
      matches: the set of potential matches (a sequence of ints)

    Yields:
      a two-tuple of (match, wrap counter). match is an int in range (1-12),
      wrapcount is a int indicating how many times we've wrapped around.
    """
        potential = matches = sorted(matches)
        after = start - 1
        wrapcount = 0
        while True:
            potential = [x for x in potential if x > after]
            if not potential:
                wrapcount += 1
                potential = matches
            after = potential[0]
            yield (after, wrapcount)

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