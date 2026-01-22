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
def _WaitForRetry(retries_left):
    """Sleeps for a period of time based on the retry count.

  Args:
    retries_left: int, The number of retries remaining.  Should be in the range
      of NUM_RETRIES - 1 to 0.
  """
    time_to_wait = 0.1 * (2 * (NUM_RETRIES - retries_left))
    logging.debug('Waiting for retry: [%s]', time_to_wait)
    time.sleep(time_to_wait)