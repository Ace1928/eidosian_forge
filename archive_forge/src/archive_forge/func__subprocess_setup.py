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
def _subprocess_setup(on_preexec_fn):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    if on_preexec_fn:
        on_preexec_fn()