from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchSessionSparkApplicationStageAttemptsResponse(_messages.Message):
    """A list of Stage Attempts for a Stage of a Spark Application.

  Fields:
    nextPageToken: This token is included in the response if there are more
      results to fetch. To fetch additional results, provide this value as the
      page_token in a subsequent
      SearchSessionSparkApplicationStageAttemptsRequest.
    sparkApplicationStageAttempts: Output only. Data corresponding to a stage
      attempts
  """
    nextPageToken = _messages.StringField(1)
    sparkApplicationStageAttempts = _messages.MessageField('StageData', 2, repeated=True)