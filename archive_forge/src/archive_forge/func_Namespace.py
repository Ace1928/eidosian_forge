from collections import deque, namedtuple
from datetime import timedelta
from celery.utils.functional import memoize
from celery.utils.serialization import strtobool
def Namespace(__old__=None, **options):
    if __old__ is not None:
        for key, opt in options.items():
            if not opt.old:
                opt.old = {o.format(key) for o in __old__}
    return options