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
class LoopingCallTimeOut(Exception):
    """Exception for a timed out LoopingCall.

    The LoopingCall will raise this exception when a timeout is provided
    and it is exceeded.
    """
    pass