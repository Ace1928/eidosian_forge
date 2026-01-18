import os
import sys
import asyncio
import threading
from uuid import uuid4
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from lazyops.models import LazyData
from lazyops.common import lazy_import, lazylibs
from lazyops.retry import retryable
def wrapper_cache(func):
    func = lru_cache(maxsize=maxsize)(func)
    func.lifetime = timedelta(seconds=seconds)
    func.expiration = datetime.utcnow() + func.lifetime

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        if datetime.utcnow() >= func.expiration:
            func.cache_clear()
            func.expiration = datetime.utcnow() + func.lifetime
        return func(*args, **kwargs)
    return wrapped_func