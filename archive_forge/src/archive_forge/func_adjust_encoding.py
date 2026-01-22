from six import PY2
from functools import wraps
from datetime import datetime, timedelta, tzinfo
@wraps(namefunc)
def adjust_encoding(*args, **kwargs):
    name = namefunc(*args, **kwargs)
    if name is not None:
        name = name.encode()
    return name