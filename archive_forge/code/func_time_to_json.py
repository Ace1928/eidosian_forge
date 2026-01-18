import re
import traitlets
import datetime as dt
def time_to_json(pyt, manager):
    """Serialize a Python time object to json."""
    if pyt is None:
        return None
    else:
        return dict(hours=pyt.hour, minutes=pyt.minute, seconds=pyt.second, milliseconds=pyt.microsecond / 1000)