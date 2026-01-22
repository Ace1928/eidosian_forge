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
def _PollUntilDoneUsingOperationWait(self, timeout_sec=_POLLING_TIMEOUT_SEC):
    """Polls the operation with operation method."""
    wait_request = self.OperationWaitRequest()
    start = time_util.CurrentTimeSec()
    while not self.IsDone():
        if time_util.CurrentTimeSec() - start > timeout_sec:
            self.errors.append((None, 'operation {} timed out'.format(self.operation.name)))
            _RecordProblems(self.operation, self.warnings, self.errors)
            return
        try:
            self.operation = self._CallService(self.operation_service.Wait, wait_request)
        except apitools_exceptions.HttpError:
            return
    _RecordProblems(self.operation, self.warnings, self.errors)