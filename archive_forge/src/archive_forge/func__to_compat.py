from collections import deque, namedtuple
from datetime import timedelta
from celery.utils.functional import memoize
from celery.utils.serialization import strtobool
def _to_compat(ns, key, opt):
    if opt.old:
        return [(oldkey.format(key).upper(), ns + key, opt) for oldkey in opt.old]
    return [((ns + key).upper(), ns + key, opt)]