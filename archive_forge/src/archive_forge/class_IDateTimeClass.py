from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import tzinfo
from zope.interface import Attribute
from zope.interface import Interface
from zope.interface import classImplements
class IDateTimeClass(Interface):
    """This is the datetime class interface.

    This is symbolic; this module does **not** make
    `datetime.datetime` provide this interface.
    """
    min = Attribute('The earliest representable datetime')
    max = Attribute('The latest representable datetime')
    resolution = Attribute('The smallest possible difference between non-equal datetime objects')

    def today():
        """Return the current local datetime, with tzinfo None.

        This is equivalent to ``datetime.fromtimestamp(time.time())``.

        .. seealso:: `now`, `fromtimestamp`.
        """

    def now(tz=None):
        """Return the current local date and time.

        If optional argument *tz* is None or not specified, this is like `today`,
        but, if possible, supplies more precision than can be gotten from going
        through a `time.time` timestamp (for example, this may be possible on
        platforms supplying the C ``gettimeofday()`` function).

        Else tz must be an instance of a class tzinfo subclass, and the current
        date and time are converted to tz's time zone. In this case the result
        is equivalent to tz.fromutc(datetime.utcnow().replace(tzinfo=tz)).

        .. seealso:: `today`, `utcnow`.
        """

    def utcnow():
        """Return the current UTC date and time, with tzinfo None.

        This is like `now`, but returns the current UTC date and time, as a
        naive datetime object.

        .. seealso:: `now`.
        """

    def fromtimestamp(timestamp, tz=None):
        """Return the local date and time corresponding to the POSIX timestamp.

        Same as is returned by time.time(). If optional argument tz is None or
        not specified, the timestamp is converted to the platform's local date
        and time, and the returned datetime object is naive.

        Else tz must be an instance of a class tzinfo subclass, and the
        timestamp is converted to tz's time zone. In this case the result is
        equivalent to
        ``tz.fromutc(datetime.utcfromtimestamp(timestamp).replace(tzinfo=tz))``.

        fromtimestamp() may raise `ValueError`, if the timestamp is out of the
        range of values supported by the platform C localtime() or gmtime()
        functions. It's common for this to be restricted to years in 1970
        through 2038. Note that on non-POSIX systems that include leap seconds
        in their notion of a timestamp, leap seconds are ignored by
        fromtimestamp(), and then it's possible to have two timestamps
        differing by a second that yield identical datetime objects.

        .. seealso:: `utcfromtimestamp`.
        """

    def utcfromtimestamp(timestamp):
        """Return the UTC datetime from the POSIX timestamp with tzinfo None.

        This may raise `ValueError`, if the timestamp is out of the range of
        values supported by the platform C ``gmtime()`` function. It's common for
        this to be restricted to years in 1970 through 2038.

        .. seealso:: `fromtimestamp`.
        """

    def fromordinal(ordinal):
        """Return the datetime from the proleptic Gregorian ordinal.

        January 1 of year 1 has ordinal 1. `ValueError` is raised unless
        1 <= ordinal <= datetime.max.toordinal().
        The hour, minute, second and microsecond of the result are all 0, and
        tzinfo is None.
        """

    def combine(date, time):
        """Return a new datetime object.

        Its date members are equal to the given date object's, and whose time
        and tzinfo members are equal to the given time object's. For any
        datetime object *d*, ``d == datetime.combine(d.date(), d.timetz())``.
        If date is a datetime object, its time and tzinfo members are ignored.
        """