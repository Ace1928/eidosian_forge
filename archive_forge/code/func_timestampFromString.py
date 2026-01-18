import os
import time
from datetime import datetime, timezone
import calendar
def timestampFromString(value):
    wkday, mnth = value[:7].split()
    t = datetime.strptime(value[7:], ' %d %H:%M:%S %Y')
    t = t.replace(month=MONTHNAMES.index(mnth), tzinfo=timezone.utc)
    wkday_idx = DAYNAMES.index(wkday)
    assert t.weekday() == wkday_idx, '"' + value + '" has inconsistent weekday'
    return int(t.timestamp()) - epoch_diff