import logging
import os
import sys
import warnings
from logging.handlers import WatchedFileHandler
from kombu.utils.encoding import set_default_encoding_file
from celery import signals
from celery._state import get_current_task
from celery.exceptions import CDeprecationWarning, CPendingDeprecationWarning
from celery.local import class_property
from celery.utils.log import (ColorFormatter, LoggingProxy, get_logger, get_multiprocessing_logger, mlevel,
from celery.utils.nodenames import node_format
from celery.utils.term import colored
def _detect_handler(self, logfile=None):
    """Create handler from filename, an open stream or `None` (stderr)."""
    logfile = sys.__stderr__ if logfile is None else logfile
    if hasattr(logfile, 'write'):
        return logging.StreamHandler(logfile)
    return WatchedFileHandler(logfile, encoding='utf-8')