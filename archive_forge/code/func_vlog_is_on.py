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
def vlog_is_on(level):
    """Checks if vlog is enabled for the given level in caller's source file.

  Args:
    level: int, the C++ verbose logging level at which to log the message,
        e.g. 1, 2, 3, 4... While absl level constants are also supported,
        callers should prefer level_debug|level_info|... calls for
        checking those.

  Returns:
    True if logging is turned on for that level.
  """
    if level > converter.ABSL_DEBUG:
        standard_level = converter.STANDARD_DEBUG - (level - 1)
    else:
        if level < converter.ABSL_FATAL:
            level = converter.ABSL_FATAL
        standard_level = converter.absl_to_standard(level)
    return _absl_logger.isEnabledFor(standard_level)