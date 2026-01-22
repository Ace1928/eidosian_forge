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
def PrintProgress(self, stream=sys.stderr):
    """Prints progress and throughput/time estimation.

    If a ProducerThreadMessage or SeekAheadMessage has been provided,
    it outputs the number of files completed, number of total files,
    the current progress, the total size, and the percentage it
    represents.
    If none of those have been provided, it only includes the number of files
    completed, the current progress and total size (which might be updated),
    with no percentage as we do not know if more files are coming.
    It may also include time estimation (available only given
    ProducerThreadMessage or SeekAheadMessage provided) and throughput. For that
    to happen, there is an extra condition of at least first_throughput_latency
    seconds having been passed since the UIController started, and that
    either the ProducerThread or the SeekAheadThread have estimated total
    number of files and total size.

    Args:
      stream: Stream to print messages. Usually sys.stderr, but customizable
              for testing.
    """
    total_remaining = self.total_size - self.total_progress
    if self.throughput:
        time_remaining = total_remaining / self.throughput
    else:
        time_remaining = None
    char_to_print = self.GetSpinner()
    if self.num_objects_source <= EstimationSource.SEEK_AHEAD_THREAD:
        objects_completed = '[' + DecimalShort(self.objects_finished) + '/' + DecimalShort(self.num_objects) + ' files]'
    else:
        objects_completed = '[' + DecimalShort(self.objects_finished) + ' files]'
    bytes_progress = '[%s/%s]' % (BytesToFixedWidthString(self.total_progress), BytesToFixedWidthString(self.total_size))
    if self.total_size_source <= EstimationSource.SEEK_AHEAD_THREAD:
        if self.num_objects == self.objects_finished:
            percentage = '100'
        else:
            percentage = '%3d' % min(99, int(100 * float(self.total_progress) / self.total_size))
        percentage_completed = percentage + '% Done'
    else:
        percentage_completed = ''
    if self.refresh_message_time - self.start_time > self.first_throughput_latency:
        throughput = BytesToFixedWidthString(self.throughput) + '/s'
        if self.total_size_source <= EstimationSource.PRODUCER_THREAD_ESTIMATE and self.throughput:
            time_remaining_str = 'ETA ' + PrettyTime(time_remaining)
        else:
            time_remaining_str = ''
    else:
        throughput = ''
        time_remaining_str = ''
    format_str = '{char_to_print} {objects_completed}{bytes_progress} {percentage_completed} {throughput} {time_remaining_str}'
    string_to_print = format_str.format(char_to_print=char_to_print, objects_completed=objects_completed, bytes_progress=bytes_progress, percentage_completed=percentage_completed, throughput=throughput, time_remaining_str=time_remaining_str)
    remaining_width = self.console_width - len(string_to_print)
    if not self.quiet_mode:
        stream.write(string_to_print + max(remaining_width, 0) * ' ' + '\r')