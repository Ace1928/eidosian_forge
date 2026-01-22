from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResultStorage(_messages.Message):
    """Locations where the results of running the test are stored.

  Fields:
    googleCloudStorage: Required.
    resultsUrl: Output only. URL to the results in the Firebase Web Console.
    toolResultsExecution: Output only. The tool results execution that results
      are written to.
    toolResultsHistory: The tool results history that contains the tool
      results execution that results are written to. If not provided, the
      service will choose an appropriate value.
  """
    googleCloudStorage = _messages.MessageField('GoogleCloudStorage', 1)
    resultsUrl = _messages.StringField(2)
    toolResultsExecution = _messages.MessageField('ToolResultsExecution', 3)
    toolResultsHistory = _messages.MessageField('ToolResultsHistory', 4)