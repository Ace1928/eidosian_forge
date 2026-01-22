from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import batch_helper
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import single_request_helper
from googlecloudsdk.api_lib.util import exceptions as http_exceptions
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _PollUntilDoneUsingOperationGet(self, timeout_sec=_POLLING_TIMEOUT_SEC):
    """Polls the operation with operation Get method."""
    get_request = self.OperationGetRequest()
    start = time_util.CurrentTimeSec()
    poll_time_interval = 0
    max_poll_interval = 5
    while True:
        if time_util.CurrentTimeSec() - start > timeout_sec:
            self.errors.append((None, 'operation {} timed out'.format(self.operation.name)))
            _RecordProblems(self.operation, self.warnings, self.errors)
            return
        try:
            self.operation = self._CallService(self.operation_service.Get, get_request)
        except apitools_exceptions.HttpError:
            return
        if self.IsDone():
            _RecordProblems(self.operation, self.warnings, self.errors)
            return
        poll_time_interval = min(poll_time_interval + 1, max_poll_interval)
        time_util.Sleep(poll_time_interval)