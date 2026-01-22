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
def LogRetryableError(message):
    """Logs that a retryable error was caught for a gsutil command.

  Args:
    message: The RetryableErrorMessage posted to the global status queue.
  """
    collector = MetricsCollector.GetCollector()
    if collector:
        collector.retryable_errors[message.error_type] += 1
        if message.is_service_error:
            LogPerformanceSummaryParams(num_retryable_service_errors=1)
        else:
            LogPerformanceSummaryParams(num_retryable_network_errors=1)