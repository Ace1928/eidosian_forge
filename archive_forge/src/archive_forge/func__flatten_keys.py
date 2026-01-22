from collections import deque, namedtuple
from datetime import timedelta
from celery.utils.functional import memoize
from celery.utils.serialization import strtobool
def _flatten_keys(ns, key, opt):
    return [(ns + key, opt)]