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
class _ThroughputInformation(object):
    """Class that contains all information needed for throughput calculation.

    This _ThroughputInformation is used to track progress and time at several
    points of our operation.
    """

    def __init__(self, progress, report_time):
        """Constructor of _ThroughputInformation.

      Args:
        progress: The current progress, in bytes/second or objects/second.
        report_time: Float representing when progress was reported (seconds
            since Epoch).
      """
        self.progress = progress
        self.time = report_time