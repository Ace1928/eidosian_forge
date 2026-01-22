from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
def _Normalize(self):
    """Normalizes duration values to integers in ISO 8601 ranges.

    Normalization makes formatted durations aesthetically pleasing. For example,
    P2H30M0.5S instead of P9000.5S. It also determines if the duration is exact
    or a calendar duration.
    """

    def _Percolate(f):
        int_value = int(f)
        fraction = round(round(f, 4) - int_value, 4)
        return (int_value, fraction)
    self.years, fraction = _Percolate(self.years)
    if fraction:
        self.days += _DAYS_PER_YEAR * fraction
    self.months, fraction = _Percolate(self.months)
    if fraction:
        self.days += int(_DAYS_PER_YEAR * fraction / _MONTHS_PER_YEAR)
    self.days, fraction = _Percolate(self.days)
    if fraction:
        self.hours += _HOURS_PER_DAY * fraction
    self.hours, fraction = _Percolate(self.hours)
    if fraction:
        self.minutes += _MINUTES_PER_HOUR * fraction
    self.minutes, fraction = _Percolate(self.minutes)
    if fraction:
        self.seconds += _SECONDS_PER_MINUTE * fraction
    self.seconds, fraction = _Percolate(self.seconds)
    if fraction:
        self.microseconds = int(_MICROSECONDS_PER_SECOND * fraction)
    self.total_seconds = 0.0
    carry = int(self.microseconds / _MICROSECONDS_PER_SECOND)
    self.microseconds -= int(carry * _MICROSECONDS_PER_SECOND)
    self.total_seconds += self.microseconds / _MICROSECONDS_PER_SECOND
    self.seconds += carry
    carry = int(self.seconds / _SECONDS_PER_MINUTE)
    self.seconds -= carry * _SECONDS_PER_MINUTE
    self.total_seconds += self.seconds
    self.minutes += carry
    carry = int(self.minutes / _MINUTES_PER_HOUR)
    self.minutes -= carry * _MINUTES_PER_HOUR
    self.total_seconds += self.minutes * _SECONDS_PER_MINUTE
    self.hours += carry
    if not self.calendar:
        if self.days or self.months or self.years:
            self.calendar = True
        else:
            self.total_seconds += self.hours * _SECONDS_PER_HOUR
            return
    carry = int(self.hours / _HOURS_PER_DAY)
    self.hours -= carry * _HOURS_PER_DAY
    self.total_seconds += self.hours * _SECONDS_PER_HOUR
    self.days += carry
    if self.days >= int(_DAYS_PER_YEAR + 1):
        self.days -= int(_DAYS_PER_YEAR + 1)
        self.years += 1
    elif self.days <= -int(_DAYS_PER_YEAR + 1):
        self.days += int(_DAYS_PER_YEAR + 1)
        self.years -= 1
    carry = int(self.days / _DAYS_PER_YEAR)
    self.days -= int(carry * _DAYS_PER_YEAR)
    self.total_seconds += self.days * _SECONDS_PER_DAY
    self.years += carry
    carry = int(self.months / _MONTHS_PER_YEAR)
    self.months -= carry * _MONTHS_PER_YEAR
    self.total_seconds += self.months * _SECONDS_PER_MONTH
    self.years += carry
    self.total_seconds += self.years * _SECONDS_PER_YEAR
    self.total_seconds = round(self.total_seconds, 0) + self.microseconds / _MICROSECONDS_PER_SECOND