from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import time
from apitools.base.py import http_wrapper
from gslib import thread_message
from gslib.utils import constants
from retry_decorator import retry_decorator
Exception handler that disables retries in apitools data transfers.

    Post a gslib.thread_message.RetryableErrorMessage to the global status
    queue. We handle the actual retries within the download and upload
    functions.

    Args:
      retry_args: An apitools ExceptionRetryArgs tuple.
    