from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatapipelinesV1ListPipelinesResponse(_messages.Message):
    """Response message for ListPipelines.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    pipelines: Results that matched the filter criteria and were accessible to
      the caller. Results are always in descending order of pipeline creation
      date.
  """
    nextPageToken = _messages.StringField(1)
    pipelines = _messages.MessageField('GoogleCloudDatapipelinesV1Pipeline', 2, repeated=True)