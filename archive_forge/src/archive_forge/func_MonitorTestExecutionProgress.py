from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import time
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def MonitorTestExecutionProgress(self, test_id):
    """Monitor and report the progress of a single running test.

    This method prints more detailed test progress messages for the case where
    the matrix has exactly one supported test configuration.

    Args:
      test_id: str, the unique id of the single supported test in the matrix.

    Raises:
      TestLabInfrastructureError if the Test service reports a backend error.

    """
    states = self._messages.TestExecution.StateValueValuesEnum
    last_state = ''
    error = ''
    progress = []
    last_progress_len = 0
    while True:
        status = self._GetTestExecutionStatus(test_id)
        timestamp = self._clock().strftime(_TIMESTAMP_FORMAT)
        details = status.testDetails
        if details:
            error = details.errorMessage or ''
            progress = details.progressMessages or []
        for msg in progress[last_progress_len:]:
            log.status.Print('{0} {1}'.format(timestamp, msg.rstrip()))
        last_progress_len = len(progress)
        if status.state == states.ERROR:
            raise exceptions.TestLabInfrastructureError(error)
        if status.state == states.UNSUPPORTED_ENVIRONMENT:
            raise exceptions.AllDimensionsIncompatibleError('Device dimensions are not compatible: {d}. Please use "gcloud firebase test android models list" to determine which device dimensions are compatible.'.format(d=_FormatInvalidDimension(status.environment)))
        if status.state != last_state:
            last_state = status.state
            log.status.Print('{0} Test is {1}'.format(timestamp, self._state_names[last_state]))
        if status.state in self._completed_execution_states:
            break
        self._SleepForStatusInterval()
    matrix = self.GetTestMatrixStatus()
    while matrix.state not in self.completed_matrix_states:
        log.debug('Matrix not yet complete, still in state: %s', matrix.state)
        self._SleepForStatusInterval()
        matrix = self.GetTestMatrixStatus()
    self._LogTestComplete(matrix.state)
    return