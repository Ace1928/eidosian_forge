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
def _HandleProgressMessage(self, status_message):
    """Handles a ProgressMessage that tracks progress of a file or component.

    Args:
      status_message: The ProgressMessage to be processed.
    """
    file_name = status_message.src_url.url_string
    file_progress = self.individual_file_progress[file_name]
    key = (status_message.component_num, status_message.dst_url)
    last_update = file_progress.dict[key] if key in file_progress.dict else (0, 0)
    status_message.processed_bytes -= last_update[1]
    file_progress.new_progress_sum += status_message.processed_bytes - last_update[0]
    self.total_progress += status_message.processed_bytes - last_update[0]
    self.new_progress += status_message.processed_bytes - last_update[0]
    file_progress.dict[key] = (status_message.processed_bytes, last_update[1])
    self.last_progress_time = status_message.time