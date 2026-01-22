from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import deque
import sys
import threading
import time
from six.moves import queue as Queue
from gslib.metrics import LogPerformanceSummaryParams
from gslib.metrics import LogRetryableError
from gslib.thread_message import FileMessage
from gslib.thread_message import FinalMessage
from gslib.thread_message import MetadataMessage
from gslib.thread_message import PerformanceSummaryMessage
from gslib.thread_message import ProducerThreadMessage
from gslib.thread_message import ProgressMessage
from gslib.thread_message import RetryableErrorMessage
from gslib.thread_message import SeekAheadMessage
from gslib.thread_message import StatusMessage
from gslib.utils import parallelism_framework_util
from gslib.utils.unit_util import DecimalShort
from gslib.utils.unit_util import HumanReadableWithDecimalPlaces
from gslib.utils.unit_util import MakeHumanReadable
from gslib.utils.unit_util import PrettyTime
def _HandleSeekAheadMessage(self, status_message, stream):
    """Handles a SeekAheadMessage.

    Args:
      status_message: The SeekAheadMessage to be processed.
      stream: Stream to print messages.
    """
    estimate_message = 'Estimated work for this command: objects: %s' % status_message.num_objects
    if status_message.size:
        estimate_message += ', total size: %s' % MakeHumanReadable(status_message.size)
        if self.total_size_source >= EstimationSource.SEEK_AHEAD_THREAD:
            self.total_size_source = EstimationSource.SEEK_AHEAD_THREAD
            self.total_size = status_message.size
    if self.num_objects_source >= EstimationSource.SEEK_AHEAD_THREAD:
        self.num_objects_source = EstimationSource.SEEK_AHEAD_THREAD
        self.num_objects = status_message.num_objects
    estimate_message += '\n'
    if not self.quiet_mode:
        stream.write(estimate_message)