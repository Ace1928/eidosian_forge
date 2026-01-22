from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import atexit
from collections import defaultdict
from functools import wraps
import logging
import os
import pickle
import platform
import re
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
import six
from six.moves import input
from six.moves import urllib
import boto
from gslib import VERSION
from gslib.metrics_tuple import Metric
from gslib.utils import system_util
from gslib.utils.unit_util import CalculateThroughput
from gslib.utils.unit_util import HumanReadableToBytes
def CaptureAndLogException(func):
    """Function decorator to capture and log any exceptions.

  This is extra insurance that analytics collection will not hinder the command
  being run upon an error.

  Args:
    func: The function to wrap.

  Returns:
    The wrapped function.
  """

    @wraps(func)
    def Wrapper(*args, **kwds):
        try:
            return func(*args, **kwds)
        except Exception as e:
            logger = logging.getLogger('metrics')
            logger.debug('Exception captured in %s during metrics collection: %s', func.__name__, e)
    return Wrapper