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
def UpdateThroughput(self, cur_time, cur_progress):
    """Updates throughput if the required period for calculation has passed.

    The throughput is calculated by taking all the progress (objects or bytes)
    processed within the last sliding_throughput_period seconds, and dividing
    that by the time period between the oldest progress time within that range
    and the last progress measurement, which are defined by oldest_progress[1]
    and last_progress_time, respectively. Among the pros of this approach,
    a connection break or a sudden change in throughput is quickly noticeable.
    Furthermore, using the last throughput measurement rather than the current
    time allows us to have a better estimation of the actual throughput.

    Args:
      cur_time: Current time to check whether or not it is time for a new
                throughput measurement.
      cur_progress: The current progress, in number of objects finished or in
                    bytes.
    """
    while len(self.old_progress) > 1 and cur_time - self.old_progress[0].time > self.sliding_throughput_period:
        self.old_progress.popleft()
    if not self.old_progress:
        return
    oldest_progress = self.old_progress[0]
    if self.last_progress_time == oldest_progress.time:
        self.throughput = 0
        return
    self.throughput = (cur_progress - oldest_progress.progress) / (self.last_progress_time - oldest_progress.time)
    self.throughput = max(0, self.throughput)