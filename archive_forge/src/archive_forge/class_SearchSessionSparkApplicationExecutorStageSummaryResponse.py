from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchSessionSparkApplicationExecutorStageSummaryResponse(_messages.Message):
    """List of Executors associated with a Spark Application Stage.

  Fields:
    nextPageToken: This token is included in the response if there are more
      results to fetch. To fetch additional results, provide this value as the
      page_token in a subsequent
      SearchSessionSparkApplicationExecutorStageSummaryRequest.
    sparkApplicationStageExecutors: Details about executors used by the
      application stage.
  """
    nextPageToken = _messages.StringField(1)
    sparkApplicationStageExecutors = _messages.MessageField('ExecutorStageSummary', 2, repeated=True)