from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import enum
import errno
import hashlib
import io
import logging
import os
import shutil
import stat
import sys
import tempfile
import time
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding as encoding_util
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import retry
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _RetryOperation(exc_info, func, args, retry_test_function=lambda func, exc_info: True):
    """Attempts to retry the failed file operation.

  Args:
    exc_info: sys.exc_info(), The current exception state.
    func: function, The function that failed.
    args: (str, ...), The tuple of args that should be passed to func when
      retrying.
    retry_test_function: The function to call to determine if a retry should be
      attempted.  Takes the function that is being retried as well as the
      current exc_info.

  Returns:
    True if the operation eventually succeeded or False if it continued to fail
    for all retries.
  """
    retries_left = NUM_RETRIES
    while retries_left > 0 and retry_test_function(func, exc_info):
        logging.debug('Retrying file system operation: %s, %s, %s, retries_left=%s', func, args, exc_info, retries_left)
        retries_left -= 1
        try:
            _WaitForRetry(retries_left)
            func(*args)
            return True
        except:
            exc_info = sys.exc_info()
    return False