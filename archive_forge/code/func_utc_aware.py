import warnings
from datetime import tzinfo, timedelta, datetime
def utc_aware(unaware):
    """decorator for adding UTC tzinfo to datetime's utcfoo methods

    Deprecated since IPython 8.19.0.
    """

    def utc_method(*args, **kwargs):
        _warn_deprecated()
        dt = unaware(*args, **kwargs)
        return dt.replace(tzinfo=UTC)
    return utc_method