from suds import UnicodeMixin
import datetime
import re
import time
def _time_from_match(match_object):
    """
    Create a time object from a regular expression match.

    Returns the time object and information whether the resulting time should
    be bumped up by one microsecond due to microsecond rounding.

    Subsecond information is rounded to microseconds due to a restriction in
    the python datetime.datetime/time implementation.

    The regular expression match is expected to be from _RE_DATETIME or
    _RE_TIME.

    @param match_object: The regular expression match.
    @type match_object: B{re}.I{MatchObject}
    @return: Time object + rounding flag.
    @rtype: tuple of B{datetime}.I{time} and bool

    """
    hour = int(match_object.group('hour'))
    minute = int(match_object.group('minute'))
    second = int(match_object.group('second'))
    subsecond = match_object.group('subsecond')
    round_up = False
    microsecond = 0
    if subsecond:
        round_up = len(subsecond) > 6 and int(subsecond[6]) >= 5
        subsecond = subsecond[:6]
        microsecond = int(subsecond + '0' * (6 - len(subsecond)))
    return (datetime.time(hour, minute, second, microsecond), round_up)