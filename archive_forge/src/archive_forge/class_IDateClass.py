from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import tzinfo
from zope.interface import Attribute
from zope.interface import Interface
from zope.interface import classImplements
class IDateClass(Interface):
    """This is the date class interface.

    This is symbolic; this module does **not** make
    `datetime.date` provide this interface.
    """
    min = Attribute('The earliest representable date')
    max = Attribute('The latest representable date')
    resolution = Attribute('The smallest difference between non-equal date objects')

    def today():
        """Return the current local time.

        This is equivalent to ``date.fromtimestamp(time.time())``"""

    def fromtimestamp(timestamp):
        """Return the local date from a POSIX timestamp (like time.time())

        This may raise `ValueError`, if the timestamp is out of the range of
        values supported by the platform C ``localtime()`` function. It's common
        for this to be restricted to years from 1970 through 2038. Note that
        on non-POSIX systems that include leap seconds in their notion of a
        timestamp, leap seconds are ignored by `fromtimestamp`.
        """

    def fromordinal(ordinal):
        """Return the date corresponding to the proleptic Gregorian ordinal.

         January 1 of year 1 has ordinal 1. `ValueError` is raised unless
         1 <= ordinal <= date.max.toordinal().

         For any date *d*, ``date.fromordinal(d.toordinal()) == d``.
         """