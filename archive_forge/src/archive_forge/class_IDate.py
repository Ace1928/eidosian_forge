from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import tzinfo
from zope.interface import Attribute
from zope.interface import Interface
from zope.interface import classImplements
class IDate(IDateClass):
    """Represents a date (year, month and day) in an idealized calendar.

    Implemented by `datetime.date`.

    Operators:

    __repr__, __str__
    __cmp__, __hash__
    __add__, __radd__, __sub__ (add/radd only with timedelta arg)
    """
    year = Attribute('Between MINYEAR and MAXYEAR inclusive.')
    month = Attribute('Between 1 and 12 inclusive')
    day = Attribute('Between 1 and the number of days in the given month of the given year.')

    def replace(year, month, day):
        """Return a date with the same value.

        Except for those members given new values by whichever keyword
        arguments are specified. For example, if ``d == date(2002, 12, 31)``, then
        ``d.replace(day=26) == date(2000, 12, 26)``.
        """

    def timetuple():
        """Return a 9-element tuple of the form returned by `time.localtime`.

        The hours, minutes and seconds are 0, and the DST flag is -1.
        ``d.timetuple()`` is equivalent to
        ``(d.year, d.month, d.day, 0, 0, 0, d.weekday(), d.toordinal() -
        date(d.year, 1, 1).toordinal() + 1, -1)``
        """

    def toordinal():
        """Return the proleptic Gregorian ordinal of the date

        January 1 of year 1 has ordinal 1. For any date object *d*,
        ``date.fromordinal(d.toordinal()) == d``.
        """

    def weekday():
        """Return the day of the week as an integer.

        Monday is 0 and Sunday is 6. For example,
        ``date(2002, 12, 4).weekday() == 2``, a Wednesday.

        .. seealso:: `isoweekday`.
        """

    def isoweekday():
        """Return the day of the week as an integer.

        Monday is 1 and Sunday is 7. For example,
        date(2002, 12, 4).isoweekday() == 3, a Wednesday.

        .. seealso:: `weekday`, `isocalendar`.
        """

    def isocalendar():
        """Return a 3-tuple, (ISO year, ISO week number, ISO weekday).

        The ISO calendar is a widely used variant of the Gregorian calendar.
        See http://www.phys.uu.nl/~vgent/calendar/isocalendar.htm for a good
        explanation.

        The ISO year consists of 52 or 53 full weeks, and where a week starts
        on a Monday and ends on a Sunday. The first week of an ISO year is the
        first (Gregorian) calendar week of a year containing a Thursday. This
        is called week number 1, and the ISO year of that Thursday is the same
        as its Gregorian year.

        For example, 2004 begins on a Thursday, so the first week of ISO year
        2004 begins on Monday, 29 Dec 2003 and ends on Sunday, 4 Jan 2004, so
        that ``date(2003, 12, 29).isocalendar() == (2004, 1, 1)`` and
        ``date(2004, 1, 4).isocalendar() == (2004, 1, 7)``.
        """

    def isoformat():
        """Return a string representing the date in ISO 8601 format.

        This is 'YYYY-MM-DD'.
        For example, ``date(2002, 12, 4).isoformat() == '2002-12-04'``.
        """

    def __str__():
        """For a date *d*, ``str(d)`` is equivalent to ``d.isoformat()``."""

    def ctime():
        """Return a string representing the date.

        For example date(2002, 12, 4).ctime() == 'Wed Dec 4 00:00:00 2002'.
        d.ctime() is equivalent to time.ctime(time.mktime(d.timetuple()))
        on platforms where the native C ctime() function
        (which `time.ctime` invokes, but which date.ctime() does not invoke)
        conforms to the C standard.
        """

    def strftime(format):
        """Return a string representing the date.

        Controlled by an explicit format string. Format codes referring to
        hours, minutes or seconds will see 0 values.
        """