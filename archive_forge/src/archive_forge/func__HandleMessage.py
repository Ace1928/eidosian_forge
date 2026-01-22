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
def _HandleMessage(self, status_message, stream, cur_time=None):
    """Processes a message, updates throughput and prints progress.

    Args:
      status_message: Message to be processed. Could be None if UIThread cannot
                      retrieve message from status_queue.
      stream: stream to print messages. Usually sys.stderr, but customizable
              for testing.
      cur_time: Message time. Used to determine if it is time to refresh
                output, or calculate throughput.
    """
    self.manager.ProcessMessage(status_message, stream)
    if self.manager.ShouldPrintProgress(cur_time):
        if self.manager.ShouldTrackThroughput(cur_time):
            self.manager.UpdateThroughput(cur_time, self.manager.GetProgress())
        self.manager.PrintProgress(stream)
        self.manager.refresh_message_time = cur_time
    if self.manager.ShouldPrintSpinner(cur_time):
        self.manager.PrintSpinner(stream)
        self.manager.refresh_spinner_time = cur_time
    if (isinstance(status_message, FinalMessage) or self.manager.final_message) and self.manager.num_objects and (not self.printed_final_message):
        self.printed_final_message = True
        LogPerformanceSummaryParams(num_objects_transferred=self.manager.num_objects)
        self.manager.PrintFinalSummaryMessage(stream)