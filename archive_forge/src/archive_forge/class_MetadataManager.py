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
class MetadataManager(StatusMessageManager):
    """Manages shared state for metadata operations.

  This manager is specific for metadata operations. Among its main functions,
  it receives incoming StatusMessages, storing all necessary data
  about the current and past states of the system necessary to display to the
  UI. It also provides methods for calculating metrics such as throughput and
  estimated time remaining. Finally, it provides methods for displaying messages
  to the UI.
  """

    def __init__(self, update_message_period=1, update_spinner_period=0.6, sliding_throughput_period=5, first_throughput_latency=10, quiet_mode=False, custom_time=None, verbose=False, console_width=80):
        """Instantiates a MetadataManager.

    See argument documentation in StatusMessageManager base class.
    """
        super(MetadataManager, self).__init__(update_message_period=update_message_period, update_spinner_period=update_spinner_period, sliding_throughput_period=sliding_throughput_period, first_throughput_latency=first_throughput_latency, quiet_mode=quiet_mode, custom_time=custom_time, verbose=verbose, console_width=console_width)

    def GetProgress(self):
        """Gets the progress for a MetadataManager.

    Returns:
      The number of finished objects.
    """
        return self.objects_finished

    def _HandleMetadataMessage(self, status_message):
        """Handles a MetadataMessage.

    Args:
      status_message: The MetadataMessage to be processed.
    """
        self.objects_finished += 1
        if self.num_objects_source >= EstimationSource.INDIVIDUAL_MESSAGES:
            self.num_objects_source = EstimationSource.INDIVIDUAL_MESSAGES
            self.num_objects += 1
        self.object_report_change = True
        self.last_progress_time = status_message.time
        if self.objects_finished == self.num_objects and self.num_objects_source == EstimationSource.PRODUCER_THREAD_FINAL:
            self.final_message = True

    def ProcessMessage(self, status_message, stream):
        """Processes a message from _MainThreadUIQueue or _UIThread.

    Args:
      status_message: The StatusMessage item to be processed.
      stream: Stream to print messages.
    """
        self.object_report_change = False
        if isinstance(status_message, SeekAheadMessage):
            self._HandleSeekAheadMessage(status_message, stream)
        elif isinstance(status_message, ProducerThreadMessage):
            self._HandleProducerThreadMessage(status_message)
        elif isinstance(status_message, MetadataMessage):
            self._HandleMetadataMessage(status_message)
        elif isinstance(status_message, RetryableErrorMessage):
            LogRetryableError(status_message)
        elif isinstance(status_message, PerformanceSummaryMessage):
            self._HandlePerformanceSummaryMessage(status_message)
        self.old_progress.append(self._ThroughputInformation(self.objects_finished, status_message.time))

    def PrintProgress(self, stream=sys.stderr):
        """Prints progress and throughput/time estimation.

    Prints total number of objects and number of finished objects with the
    percentage of work done, potentially including the throughput
    (in objects/second) and estimated time remaining.

    Args:
      stream: Stream to print messages. Usually sys.stderr, but customizable
              for testing.
    """
        total_remaining = self.num_objects - self.objects_finished
        if self.throughput:
            time_remaining = total_remaining / self.throughput
        else:
            time_remaining = None
        char_to_print = self.GetSpinner()
        if self.num_objects_source <= EstimationSource.SEEK_AHEAD_THREAD:
            objects_completed = '[' + DecimalShort(self.objects_finished) + '/' + DecimalShort(self.num_objects) + ' objects]'
            if self.num_objects == self.objects_finished:
                percentage = '100'
            else:
                percentage = '%3d' % min(99, int(100 * float(self.objects_finished) / self.num_objects))
            percentage_completed = percentage + '% Done'
        else:
            objects_completed = '[' + DecimalShort(self.objects_finished) + ' objects]'
            percentage_completed = ''
        if self.refresh_message_time - self.start_time > self.first_throughput_latency:
            throughput = '%.2f objects/s' % self.throughput
            if self.num_objects_source <= EstimationSource.PRODUCER_THREAD_ESTIMATE and self.throughput:
                time_remaining_str = 'ETA ' + PrettyTime(time_remaining)
            else:
                time_remaining_str = ''
        else:
            throughput = ''
            time_remaining_str = ''
        format_str = '{char_to_print} {objects_completed} {percentage_completed} {throughput} {time_remaining_str}'
        string_to_print = format_str.format(char_to_print=char_to_print, objects_completed=objects_completed, percentage_completed=percentage_completed, throughput=throughput, time_remaining_str=time_remaining_str)
        remaining_width = self.console_width - len(string_to_print)
        if not self.quiet_mode:
            stream.write(string_to_print + max(remaining_width, 0) * ' ' + '\r')

    def CanHandleMessage(self, status_message):
        """Determines whether this manager is suitable for handling status_message.

    Args:
      status_message: The StatusMessage object to be analyzed.
    Returns:
      True if this message can be properly handled by this manager,
      False otherwise.
    """
        if isinstance(status_message, (SeekAheadMessage, ProducerThreadMessage, MetadataMessage, FinalMessage, RetryableErrorMessage, PerformanceSummaryMessage)):
            return True
        return False