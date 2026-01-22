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
@CaptureAndLogException
def LogFatalError(exception):
    """Logs that a fatal error was caught for a gsutil command.

  Args:
    exception: The exception that the command failed on.
  """
    collector = MetricsCollector.GetCollector()
    if collector:
        collector.ExtendGAParams({_GA_LABEL_MAP['Fatal Error']: exception.__class__.__name__})