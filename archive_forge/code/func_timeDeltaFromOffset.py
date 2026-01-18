from __future__ import annotations
from os import environ
from datetime import datetime, timedelta
from time import mktime as mktime_real
from twisted.python._tzhelper import FixedOffsetTimeZone
from twisted.trial.unittest import SkipTest, TestCase
def timeDeltaFromOffset(offset: str) -> timedelta:
    assert len(offset) == 5
    sign = offset[0]
    hours = int(offset[1:3])
    minutes = int(offset[3:5])
    if sign == '-':
        hours = -hours
        minutes = -minutes
    else:
        assert sign == '+'
    return timedelta(hours=hours, minutes=minutes)