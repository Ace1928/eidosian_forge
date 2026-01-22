from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import datetime
import enum
import threading
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import metrics_util
from googlecloudsdk.command_lib.storage import thread_messages
from googlecloudsdk.core import log
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import scaled_integer
import six
class _IntegerStatusTracker(_StatusTracker):
    """See super class. Tracks both file count and byte amount."""

    def __init__(self):
        super(_IntegerStatusTracker, self).__init__()
        self._completed = 0
        self._total_estimation = 0

    def _get_status_string(self):
        """See super class."""
        if self._total_estimation:
            file_progress_string = '{}/{}'.format(self._completed, self._total_estimation)
        else:
            file_progress_string = self._completed
        return 'Completed {}\r'.format(file_progress_string)

    def add_message(self, status_message):
        """See super class."""
        if isinstance(status_message, thread_messages.WorkloadEstimatorMessage):
            self._total_estimation += status_message.item_count
        elif isinstance(status_message, thread_messages.IncrementProgressMessage):
            self._completed += 1