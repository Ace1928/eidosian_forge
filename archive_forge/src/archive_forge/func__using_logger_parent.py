import logging
import numbers
import os
import sys
import threading
import traceback
from contextlib import contextmanager
from typing import AnyStr, Sequence  # noqa
from kombu.log import LOG_LEVELS
from kombu.log import get_logger as _get_logger
from kombu.utils.encoding import safe_str
from .term import colored
def _using_logger_parent(parent_logger, logger_):
    if not logger_isa(logger_, parent_logger):
        logger_.parent = parent_logger
    return logger_