from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def FetchMatrixRollupOutcome(self):
    """Gets a test execution's rolled-up outcome from the ToolResults service.

    Returns:
      The rolled-up test execution outcome (type: toolresults_v1beta3.Outcome).

    Raises:
      HttpException if the ToolResults service reports a back-end error.
    """
    request = self._messages.ToolresultsProjectsHistoriesExecutionsGetRequest(projectId=self._project, historyId=self._history_id, executionId=self._execution_id)
    try:
        response = self._client.projects_histories_executions.Get(request)
        return response.outcome
    except apitools_exceptions.HttpError as error:
        msg = 'Http error fetching test roll-up outcome: ' + util.GetError(error)
        raise exceptions.HttpException(msg)