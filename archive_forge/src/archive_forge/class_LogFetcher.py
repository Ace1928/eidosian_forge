from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import time
import enum
from googlecloudsdk.api_lib.logging import common as logging_common
from googlecloudsdk.core import log
from googlecloudsdk.core.util import times
class LogFetcher(object):
    """A class which fetches job logs."""

    class _Tasks(enum.Enum):
        POLL = 1
        CHECK_CONTINUE = 2
    LOG_BATCH_SIZE = 1000

    def __init__(self, filters=None, polling_interval=10, continue_func=lambda x: True, continue_interval=None, num_prev_entries=None):
        """Initializes the LogFetcher.

    Args:
      filters: list of string filters used in the API call.
      polling_interval: amount of time to sleep between each poll.
      continue_func: One-arg function that takes in the number of empty polls
        and outputs a boolean to decide if we should keep polling or not. If not
        given, keep polling indefinitely.
      continue_interval: int, how often to check whether the job is complete
        using continue_function. If not provided, defaults to the same as the
        polling interval.
      num_prev_entries: int, if provided, will first perform a decending
        query to set a lower bound timestamp equal to that of the n:th entry.
    """
        self.base_filters = filters or []
        self.polling_interval = polling_interval
        self.continue_interval = continue_interval or polling_interval
        self.should_continue = continue_func
        start_timestamp = _GetTailStartingTimestamp(filters, num_prev_entries)
        log.debug('start timestamp: {}'.format(start_timestamp))
        self.log_position = LogPosition(timestamp=start_timestamp)

    def GetLogs(self):
        """Retrieves a batch of logs.

    After we fetch the logs, we ensure that none of the logs have been seen
    before.  Along the way, we update the most recent timestamp.

    Returns:
      A list of valid log entries.
    """
        utcnow = datetime.datetime.utcnow()
        lower_filter = self.log_position.GetFilterLowerBound()
        upper_filter = self.log_position.GetFilterUpperBound(utcnow)
        new_filter = self.base_filters + [lower_filter, upper_filter]
        entries = logging_common.FetchLogs(log_filter=' AND '.join(new_filter), order_by='ASC', limit=self.LOG_BATCH_SIZE)
        return [entry for entry in entries if self.log_position.Update(entry.timestamp, entry.insertId)]

    def YieldLogs(self):
        """Polls Get API for more logs.

    We poll so long as our continue function, which considers the number of
    periods without new logs, returns True.

    Yields:
        A single log entry.
    """
        timer = _TaskIntervalTimer({self._Tasks.POLL: self.polling_interval, self._Tasks.CHECK_CONTINUE: self.continue_interval})
        empty_polls = 0
        tasks = [self._Tasks.POLL, self._Tasks.CHECK_CONTINUE]
        while True:
            if self._Tasks.POLL in tasks:
                logs = self.GetLogs()
                if logs:
                    empty_polls = 0
                    for log_entry in logs:
                        yield log_entry
                else:
                    empty_polls += 1
            if self._Tasks.CHECK_CONTINUE in tasks:
                should_continue = self.should_continue(empty_polls)
                if not should_continue:
                    break
            tasks = timer.Wait()