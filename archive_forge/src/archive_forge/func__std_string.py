from datetime import datetime
from struct import unpack, calcsize
from pytz.tzinfo import StaticTzInfo, DstTzInfo, memorized_ttinfo
from pytz.tzinfo import memorized_datetime, memorized_timedelta
def _std_string(s):
    """Cast a string or byte string to an ASCII string."""
    return str(s.decode('ASCII'))