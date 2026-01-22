from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import time
import enum
from googlecloudsdk.api_lib.logging import common as logging_common
from googlecloudsdk.core import log
from googlecloudsdk.core.util import times
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