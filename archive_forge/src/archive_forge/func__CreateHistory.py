from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _CreateHistory(self, history_name):
    """Creates a History using the Tool Results API.

    Args:
       history_name: string containing the name of the history to create.

    Returns:
      The history id of the created history.

    Raises:
      HttpException if the Tool Results service reports a backend error.
    """
    history = self._messages.History(name=history_name, displayName=history_name)
    request = self._messages.ToolresultsProjectsHistoriesCreateRequest(projectId=self._project, history=history)
    try:
        response = self._client.projects_histories.Create(request)
        log.debug('\nToolResultsHistories.Create response:\n{0}\n'.format(response))
        return response
    except apitools_exceptions.HttpError as error:
        msg = 'Http error while creating a Tool Results History:\n{0}'.format(util.GetError(error))
        raise exceptions.HttpException(msg)