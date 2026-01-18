import warnings
from datetime import tzinfo, timedelta, datetime
def utc_method(*args, **kwargs):
    _warn_deprecated()
    dt = unaware(*args, **kwargs)
    return dt.replace(tzinfo=UTC)