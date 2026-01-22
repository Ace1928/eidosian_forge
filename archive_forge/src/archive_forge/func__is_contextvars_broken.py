import base64
import json
import linecache
import logging
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import namedtuple
from copy import copy
from decimal import Decimal
from numbers import Real
from datetime import datetime
from functools import partial
import sentry_sdk
from sentry_sdk._compat import PY2, PY33, PY37, implements_str, text_type, urlparse
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import DEFAULT_MAX_VALUE_LENGTH
def _is_contextvars_broken():
    """
    Returns whether gevent/eventlet have patched the stdlib in a way where thread locals are now more "correct" than contextvars.
    """
    try:
        import gevent
        from gevent.monkey import is_object_patched
        version_tuple = tuple([int(part) for part in re.split('a|b|rc|\\.', gevent.__version__)[:2]])
        if is_object_patched('threading', 'local'):
            if sys.version_info >= (3, 7) and version_tuple >= (20, 9) or is_object_patched('contextvars', 'ContextVar'):
                return False
            return True
    except ImportError:
        pass
    try:
        import greenlet
        from eventlet.patcher import is_monkey_patched
        greenlet_version = parse_version(greenlet.__version__)
        if greenlet_version is None:
            logger.error('Internal error in Sentry SDK: Could not parse Greenlet version from greenlet.__version__.')
            return False
        if is_monkey_patched('thread') and greenlet_version < (0, 5):
            return True
    except ImportError:
        pass
    return False