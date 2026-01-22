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
class UIController(object):
    """Controller UI class to integrate _MainThreadUIQueue and _UIThread.

  This class receives messages from _MainThreadUIQueue and _UIThread and send
  them to an appropriate manager, which will then processes and store data about
  them.
  """

    def __init__(self, update_message_period=1, update_spinner_period=0.6, sliding_throughput_period=5, first_throughput_latency=10, quiet_mode=False, custom_time=None, verbose=False, dump_status_messages_file=None):
        """Instantiates a UIController.

    Args:
      update_message_period: Minimum period for refreshing and  displaying
          new information. A non-positive value will ignore any time
          restrictions imposed by this field.
      update_spinner_period: Minimum period for refreshing and displaying the
          spinner. A non-positive value will ignore any time restrictions
          imposed by this field.
      sliding_throughput_period: Sliding period for throughput calculation. A
          non-positive value will make it impossible to calculate the
          throughput.
      first_throughput_latency: Minimum waiting time before actually displaying
          throughput info. A non-positive value will ignore any time
          restrictions imposed by this field.
      quiet_mode: If True, do not print status messages (but still process
          them for analytics reporting as necessary).
      custom_time: If a custom start_time is desired. Used for testing.
      verbose: Tells whether or not the operation is on verbose mode.
      dump_status_messages_file: File path for logging all received status
          messages, for debugging purposes.
    """
        self.verbose = verbose
        self.update_message_period = update_message_period
        self.update_spinner_period = update_spinner_period
        self.sliding_throughput_period = sliding_throughput_period
        self.first_throughput_latency = first_throughput_latency
        self.manager = None
        self.quiet_mode = quiet_mode
        self.custom_time = custom_time
        self.console_width = 80
        self.early_estimation_messages = []
        self.printed_final_message = False
        self.dump_status_message_fp = None
        if dump_status_messages_file:
            self.dump_status_message_fp = open(dump_status_messages_file, 'ab')

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

    def Call(self, status_message, stream, cur_time=None):
        """Coordinates UI manager and calls appropriate function to handle message.

    Args:
      status_message: Message to be processed. Could be None if UIThread cannot
                      retrieve message from status_queue.
      stream: Stream to print messages. Usually sys.stderr, but customizable
              for testing.
      cur_time: Message time. Used to determine if it is time to refresh
                output, or calculate throughput.
    """
        if not isinstance(status_message, StatusMessage):
            if status_message == _ZERO_TASKS_TO_DO_ARGUMENT and (not self.manager):
                self.manager = DataManager(update_message_period=self.update_message_period, update_spinner_period=self.update_spinner_period, sliding_throughput_period=self.sliding_throughput_period, first_throughput_latency=self.first_throughput_latency, quiet_mode=self.quiet_mode, custom_time=self.custom_time, verbose=self.verbose, console_width=self.console_width)
                for estimation_message in self.early_estimation_messages:
                    self._HandleMessage(estimation_message, stream, cur_time=estimation_message.time)
            return
        if self.dump_status_message_fp:
            self.dump_status_message_fp.write(str(status_message))
            self.dump_status_message_fp.write('\n')
        if not cur_time:
            cur_time = status_message.time
        if not self.manager:
            if isinstance(status_message, SeekAheadMessage) or isinstance(status_message, ProducerThreadMessage):
                self.early_estimation_messages.append(status_message)
                return
            elif isinstance(status_message, MetadataMessage):
                self.manager = MetadataManager(update_message_period=self.update_message_period, update_spinner_period=self.update_spinner_period, sliding_throughput_period=self.sliding_throughput_period, first_throughput_latency=self.first_throughput_latency, quiet_mode=self.quiet_mode, custom_time=self.custom_time, verbose=self.verbose, console_width=self.console_width)
                for estimation_message in self.early_estimation_messages:
                    self._HandleMessage(estimation_message, stream, cur_time)
            else:
                self.manager = DataManager(update_message_period=self.update_message_period, update_spinner_period=self.update_spinner_period, sliding_throughput_period=self.sliding_throughput_period, first_throughput_latency=self.first_throughput_latency, quiet_mode=self.quiet_mode, custom_time=self.custom_time, verbose=self.verbose, console_width=self.console_width)
                for estimation_message in self.early_estimation_messages:
                    self._HandleMessage(estimation_message, stream, cur_time)
        if not self.manager.CanHandleMessage(status_message):
            if isinstance(status_message, FileMessage) or isinstance(status_message, ProgressMessage):
                self.manager = DataManager(update_message_period=self.update_message_period, update_spinner_period=self.update_spinner_period, sliding_throughput_period=self.sliding_throughput_period, first_throughput_latency=self.first_throughput_latency, custom_time=self.custom_time, verbose=self.verbose, console_width=self.console_width)
                for estimation_message in self.early_estimation_messages:
                    self._HandleMessage(estimation_message, stream, cur_time)
            else:
                return
        self._HandleMessage(status_message, stream, cur_time)