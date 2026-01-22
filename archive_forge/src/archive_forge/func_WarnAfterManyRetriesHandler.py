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
def WarnAfterManyRetriesHandler(retry_args):
    """Exception handler for http failures in apitools.

    If the user has had to wait several seconds since their first request, print
    a progress message to the terminal to let them know we're still retrying,
    then perform the default retry logic and post a
    gslib.thread_message.RetryableErrorMessage to the global status queue.

    Args:
      retry_args: An apitools ExceptionRetryArgs tuple.
    """
    if retry_args.total_wait_sec is not None and retry_args.total_wait_sec >= constants.LONG_RETRY_WARN_SEC:
        logging.info('Retrying request, attempt #%d...', retry_args.num_retries)
    if status_queue:
        status_queue.put(thread_message.RetryableErrorMessage(retry_args.exc, time.time(), num_retries=retry_args.num_retries, total_wait_sec=retry_args.total_wait_sec))
    http_wrapper.HandleExceptionsAndRebuildHttpConnections(retry_args)