from datetime import (
import time
import unittest
class FixedNoDst(tzinfo):
    """A timezone info with fixed offset, not DST"""

    def utcoffset(self, dt):
        return timedelta(hours=2, minutes=30)

    def dst(self, dt):
        return None