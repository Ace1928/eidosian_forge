import calendar
from datetime import datetime, timedelta
from twisted.python.compat import nativeString
from twisted.python.util import FancyStrMixin
def toRFC4034DateString(self):
    """
        Calculate a date by treating the current L{SerialNumber} value as a UNIX
        timestamp and return a date string in the format described in
        U{RFC4034 3.2<https://tools.ietf.org/html/rfc4034#section-3.2>}.

        @return: The date string.
        """
    d = datetime(1970, 1, 1) + timedelta(seconds=self._number)
    return nativeString(d.strftime(RFC4034_TIME_FORMAT))