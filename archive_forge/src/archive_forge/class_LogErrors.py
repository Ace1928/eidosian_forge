import functools
import logging
import multiprocessing
import os
import random
import shlex
import signal
import sys
import time
import warnings
import enum
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_concurrency._i18n import _
@enum.unique
class LogErrors(enum.IntEnum):
    """Enumerations that affect if stdout and stderr are logged on error.

    .. versionadded:: 2.7
    """
    DEFAULT = 0
    ALL = 1
    FINAL = 2