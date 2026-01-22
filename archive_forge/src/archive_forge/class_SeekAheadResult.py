from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import threading
import time
from gslib import thread_message
from gslib.utils import constants
from gslib.utils import parallelism_framework_util
class SeekAheadResult(object):
    """Result class for seek_ahead_iterator results.

  A class is used instead of a namedtuple, making it easier to document
  and use default keyword arguments.
  """

    def __init__(self, est_num_ops=1, data_bytes=0):
        """Create a SeekAheadResult.

    Args:
      est_num_ops: Number of operations the iterated result represents.
          Operation is loosely defined as a single API call for a single
          object. The total number of API calls may not be known at the time of
          iteration, so this number is approximate.
      data_bytes: Number of data bytes that will be transferred (uploaded,
          downloaded, or rewritten) for this iterated result.
    """
        self.est_num_ops = est_num_ops
        self.data_bytes = data_bytes