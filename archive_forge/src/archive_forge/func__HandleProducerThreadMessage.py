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
def _HandleProducerThreadMessage(self, status_message):
    """Handles a ProducerThreadMessage.

    Args:
      status_message: The ProducerThreadMessage to be processed.
    """
    if status_message.finished:
        if self.num_objects_source >= EstimationSource.PRODUCER_THREAD_FINAL:
            self.num_objects_source = EstimationSource.PRODUCER_THREAD_FINAL
            self.num_objects = status_message.num_objects
        if self.total_size_source >= EstimationSource.PRODUCER_THREAD_FINAL and status_message.size:
            self.total_size_source = EstimationSource.PRODUCER_THREAD_FINAL
            self.total_size = status_message.size
        return
    if self.num_objects_source >= EstimationSource.PRODUCER_THREAD_ESTIMATE:
        self.num_objects_source = EstimationSource.PRODUCER_THREAD_ESTIMATE
        self.num_objects = status_message.num_objects
    if self.total_size_source >= EstimationSource.PRODUCER_THREAD_ESTIMATE and status_message.size:
        self.total_size_source = EstimationSource.PRODUCER_THREAD_ESTIMATE
        self.total_size = status_message.size