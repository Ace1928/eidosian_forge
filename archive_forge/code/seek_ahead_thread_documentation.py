from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import threading
import time
from gslib import thread_message
from gslib.utils import constants
from gslib.utils import parallelism_framework_util
Initializes the seek ahead thread.

    Args:
      seek_ahead_iterator: Iterator matching the ProducerThread's args_iterator,
          but returning only object name and/or size in the result.
      cancel_event: threading.Event for signaling the
          seek-ahead iterator to terminate.
      status_queue: Status queue for posting summary of fully iterated results.
    