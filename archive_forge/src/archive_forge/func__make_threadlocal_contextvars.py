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
def _make_threadlocal_contextvars(local):

    class ContextVar(object):

        def __init__(self, name, default=None):
            self._name = name
            self._default = default
            self._local = local()
            self._original_local = local()

        def get(self, default=None):
            return getattr(self._local, 'value', default or self._default)

        def set(self, value):
            token = str(random.getrandbits(64))
            original_value = self.get()
            setattr(self._original_local, token, original_value)
            self._local.value = value
            return token

        def reset(self, token):
            self._local.value = getattr(self._original_local, token)
            del self._original_local[token]
    return ContextVar