import calendar
from datetime import datetime, timedelta
from twisted.python.compat import nativeString
from twisted.python.util import FancyStrMixin
def _convertOther(self, other: object) -> 'SerialNumber':
    """
        Check that a foreign object is suitable for use in the comparison or
        arithmetic magic methods of this L{SerialNumber} instance. Raise
        L{TypeError} if not.

        @param other: The foreign L{object} to be checked.
        @return: C{other} after compatibility checks and possible coercion.
        @raise TypeError: If C{other} is not compatible.
        """
    if not isinstance(other, SerialNumber):
        raise TypeError(f'cannot compare or combine {self!r} and {other!r}')
    if self._serialBits != other._serialBits:
        raise TypeError('cannot compare or combine SerialNumber instances with different serialBits. %r and %r' % (self, other))
    return other