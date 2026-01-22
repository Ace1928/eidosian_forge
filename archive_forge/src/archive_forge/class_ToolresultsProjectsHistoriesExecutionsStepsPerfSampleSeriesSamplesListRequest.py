from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolresultsProjectsHistoriesExecutionsStepsPerfSampleSeriesSamplesListRequest(_messages.Message):
    """A ToolresultsProjectsHistoriesExecutionsStepsPerfSampleSeriesSamplesList
  Request object.

  Fields:
    executionId: A tool results execution ID.
    historyId: A tool results history ID.
    pageSize: The default page size is 500 samples, and the maximum size is
      5000. If the page_size is greater than 5000, the effective page size
      will be 5000
    pageToken: Optional, the next_page_token returned in the previous response
    projectId: The cloud project
    sampleSeriesId: A sample series id
    stepId: A tool results step ID.
  """
    executionId = _messages.StringField(1, required=True)
    historyId = _messages.StringField(2, required=True)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    projectId = _messages.StringField(5, required=True)
    sampleSeriesId = _messages.StringField(6, required=True)
    stepId = _messages.StringField(7, required=True)