from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import tzinfo
from zope.interface import Attribute
from zope.interface import Interface
from zope.interface import classImplements
class IDateTime(IDate, IDateTimeClass):
    """Object contains all the information from a date object and a time object.

    Implemented by `datetime.datetime`.
    """
    year = Attribute('Year between MINYEAR and MAXYEAR inclusive')
    month = Attribute('Month between 1 and 12 inclusive')
    day = Attribute('Day between 1 and the number of days in the given month of the year')
    hour = Attribute('Hour in range(24)')
    minute = Attribute('Minute in range(60)')
    second = Attribute('Second in range(60)')
    microsecond = Attribute('Microsecond in range(1000000)')
    tzinfo = Attribute('The object passed as the tzinfo argument to the datetime constructor\n        or None if none was passed')

    def date():
        """Return date object with same year, month and day."""

    def time():
        """Return time object with same hour, minute, second, microsecond.

        tzinfo is None.

        .. seealso:: Method :meth:`timetz`.
        """

    def timetz():
        """Return time object with same hour, minute, second, microsecond,
        and tzinfo.

        .. seealso:: Method :meth:`time`.
        """

    def replace(year, month, day, hour, minute, second, microsecond, tzinfo):
        """Return a datetime with the same members, except for those members
        given new values by whichever keyword arguments are specified.

        Note that ``tzinfo=None`` can be specified to create a naive datetime from
        an aware datetime with no conversion of date and time members.
        """

    def astimezone(tz):
        """Return a datetime object with new tzinfo member tz, adjusting the
        date and time members so the result is the same UTC time as self, but
        in tz's local time.

        tz must be an instance of a tzinfo subclass, and its utcoffset() and
        dst() methods must not return None. self must be aware (self.tzinfo
        must not be None, and self.utcoffset() must not return None).

        If self.tzinfo is tz, self.astimezone(tz) is equal to self: no
        adjustment of date or time members is performed. Else the result is
        local time in time zone tz, representing the same UTC time as self:

            after astz = dt.astimezone(tz), astz - astz.utcoffset()

        will usually have the same date and time members as dt - dt.utcoffset().
        The discussion of class `datetime.tzinfo` explains the cases at Daylight Saving
        Time transition boundaries where this cannot be achieved (an issue only
        if tz models both standard and daylight time).

        If you merely want to attach a time zone object *tz* to a datetime *dt*
        without adjustment of date and time members, use ``dt.replace(tzinfo=tz)``.
        If you merely want to remove the time zone object from an aware
        datetime dt without conversion of date and time members, use
        ``dt.replace(tzinfo=None)``.

        Note that the default `tzinfo.fromutc` method can be overridden in a
        tzinfo subclass to effect the result returned by `astimezone`.
        """

    def utcoffset():
        """Return the timezone offset in minutes east of UTC (negative west of
        UTC)."""

    def dst():
        """Return 0 if DST is not in effect, or the DST offset (in minutes
        eastward) if DST is in effect.
        """

    def tzname():
        """Return the timezone name."""

    def timetuple():
        """Return a 9-element tuple of the form returned by `time.localtime`."""

    def utctimetuple():
        """Return UTC time tuple compatilble with `time.gmtime`."""

    def toordinal():
        """Return the proleptic Gregorian ordinal of the date.

        The same as self.date().toordinal().
        """

    def weekday():
        """Return the day of the week as an integer.

        Monday is 0 and Sunday is 6. The same as self.date().weekday().
        See also isoweekday().
        """

    def isoweekday():
        """Return the day of the week as an integer.

        Monday is 1 and Sunday is 7. The same as self.date().isoweekday.

        .. seealso:: `weekday`, `isocalendar`.
        """

    def isocalendar():
        """Return a 3-tuple, (ISO year, ISO week number, ISO weekday).

        The same as self.date().isocalendar().
        """

    def isoformat(sep='T'):
        """Return a string representing the date and time in ISO 8601 format.

        YYYY-MM-DDTHH:MM:SS.mmmmmm or YYYY-MM-DDTHH:MM:SS if microsecond is 0

        If `utcoffset` does not return None, a 6-character string is appended,
        giving the UTC offset in (signed) hours and minutes:

        YYYY-MM-DDTHH:MM:SS.mmmmmm+HH:MM or YYYY-MM-DDTHH:MM:SS+HH:MM
        if microsecond is 0.

        The optional argument sep (default 'T') is a one-character separator,
        placed between the date and time portions of the result.
        """

    def __str__():
        """For a datetime instance *d*, ``str(d)`` is equivalent to ``d.isoformat(' ')``.
        """

    def ctime():
        """Return a string representing the date and time.

        ``datetime(2002, 12, 4, 20, 30, 40).ctime() == 'Wed Dec 4 20:30:40 2002'``.
        ``d.ctime()`` is equivalent to ``time.ctime(time.mktime(d.timetuple()))`` on
        platforms where the native C ``ctime()`` function (which `time.ctime`
        invokes, but which `datetime.ctime` does not invoke) conforms to the
        C standard.
        """

    def strftime(format):
        """Return a string representing the date and time.

        This is controlled by an explicit format string.
        """