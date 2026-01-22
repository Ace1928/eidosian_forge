import functools
import random
import sys
import time
from eventlet import event
from eventlet import greenthread
from oslo_log import log as logging
from oslo_utils import eventletutils
from oslo_utils import excutils
from oslo_utils import reflection
from oslo_utils import timeutils
from oslo_service._i18n import _
def _safe_wrapper(f, kind, func_name):
    """Wrapper that calls into wrapped function and logs errors as needed."""

    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except LoopingCallDone:
            raise
        except Exception:
            LOG.error('%(kind)s %(func_name)r failed', {'kind': kind, 'func_name': func_name}, exc_info=True)
            return 0
    return func