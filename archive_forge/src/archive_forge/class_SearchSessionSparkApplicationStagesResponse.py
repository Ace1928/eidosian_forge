from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchSessionSparkApplicationStagesResponse(_messages.Message):
    """A list of stages associated with a Spark Application.

  Fields:
    nextPageToken: This token is included in the response if there are more
      results to fetch. To fetch additional results, provide this value as the
      page_token in a subsequent SearchSessionSparkApplicationStages.
    sparkApplicationStages: Output only. Data corresponding to a stage.
  """
    nextPageToken = _messages.StringField(1)
    sparkApplicationStages = _messages.MessageField('StageData', 2, repeated=True)