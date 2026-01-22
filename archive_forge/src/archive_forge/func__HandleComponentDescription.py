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
def _HandleComponentDescription(self, status_message):
    """Handles a FileMessage that describes a component.

    Args:
      status_message: The FileMessage to be processed.
    """
    if status_message.message_type == FileMessage.EXISTING_COMPONENT and (not status_message.finished):
        self.existing_components += 1
        file_name = status_message.src_url.url_string
        file_progress = self.individual_file_progress[file_name]
        key = (status_message.component_num, status_message.dst_url)
        file_progress.dict[key] = (0, status_message.size)
        file_progress.existing_progress_sum += status_message.size
        self.total_progress += status_message.size
        self.existing_progress += status_message.size
    elif status_message.message_type == FileMessage.COMPONENT_TO_UPLOAD or status_message.message_type == FileMessage.COMPONENT_TO_DOWNLOAD:
        if not status_message.finished:
            self.component_total += 1
            if status_message.message_type == FileMessage.COMPONENT_TO_DOWNLOAD:
                file_name = status_message.src_url.url_string
                file_progress = self.individual_file_progress[file_name]
                file_progress.existing_progress_sum += status_message.bytes_already_downloaded
                key = (status_message.component_num, status_message.dst_url)
                file_progress.dict[key] = (0, status_message.bytes_already_downloaded)
                self.total_progress += status_message.bytes_already_downloaded
                self.existing_progress += status_message.bytes_already_downloaded
        else:
            self.finished_components += 1
            file_name = status_message.src_url.url_string
            file_progress = self.individual_file_progress[file_name]
            key = (status_message.component_num, status_message.dst_url)
            last_update = file_progress.dict[key] if key in file_progress.dict else (0, 0)
            self.total_progress += status_message.size - sum(last_update)
            self.new_progress += status_message.size - sum(last_update)
            self.last_progress_time = status_message.time
            file_progress.new_progress_sum += status_message.size - sum(last_update)
            file_progress.dict[key] = (status_message.size - last_update[1], last_update[1])