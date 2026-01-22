from __future__ import absolute_import
import email
import logging
import re
import time
import warnings
from collections import namedtuple
from itertools import takewhile
from ..exceptions import (
from ..packages import six
@BACKOFF_MAX.setter
def BACKOFF_MAX(cls, value):
    warnings.warn("Using 'Retry.BACKOFF_MAX' is deprecated and will be removed in v2.0. Use 'Retry.DEFAULT_BACKOFF_MAX' instead", DeprecationWarning)
    cls.DEFAULT_BACKOFF_MAX = value