import time
from functools import wraps
from typing import Any, Dict, Tuple
from jedi import settings
from parso.cache import parser_cache
def _temp(key_func):
    dct = {}
    _time_caches[time_add_setting] = dct

    def wrapper(*args, **kwargs):
        generator = key_func(*args, **kwargs)
        key = next(generator)
        try:
            expiry, value = dct[key]
            if expiry > time.time():
                return value
        except KeyError:
            pass
        value = next(generator)
        time_add = getattr(settings, time_add_setting)
        if key is not None:
            dct[key] = (time.time() + time_add, value)
        return value
    return wrapper