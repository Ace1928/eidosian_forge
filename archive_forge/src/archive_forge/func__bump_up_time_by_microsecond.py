from suds import UnicodeMixin
import datetime
import re
import time
def _bump_up_time_by_microsecond(time):
    """
    Helper function bumping up the given datetime.time by a microsecond,
    cycling around silently to 00:00:00.0 in case of an overflow.

    @param time: Time object.
    @type time: B{datetime}.I{time}
    @return: Time object.
    @rtype: B{datetime}.I{time}

    """
    dt = datetime.datetime(2000, 1, 1, time.hour, time.minute, time.second, time.microsecond)
    dt += datetime.timedelta(microseconds=1)
    return dt.time()