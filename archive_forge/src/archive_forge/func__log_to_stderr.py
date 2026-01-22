from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import getpass
import io
import itertools
import logging
import os
import socket
import struct
import sys
import time
import timeit
import traceback
import types
import warnings
from absl import flags
from absl._collections_abc import abc
from absl.logging import converter
import six
def _log_to_stderr(self, record):
    """Emits the record to stderr.

    This temporarily sets the handler stream to stderr, calls
    StreamHandler.emit, then reverts the stream back.

    Args:
      record: logging.LogRecord, the record to log.
    """
    old_stream = self.stream
    self.stream = sys.stderr
    try:
        super(PythonHandler, self).emit(record)
    finally:
        self.stream = old_stream