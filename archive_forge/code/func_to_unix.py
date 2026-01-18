from collections import namedtuple
import datetime
import sys
import struct
def to_unix(self):
    """Get the timestamp as a floating-point value.

        :returns: posix timestamp
        :rtype: float
        """
    return self.seconds + self.nanoseconds / 1000000000.0