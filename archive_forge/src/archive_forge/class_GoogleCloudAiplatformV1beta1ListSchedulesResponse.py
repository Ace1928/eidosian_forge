from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListSchedulesResponse(_messages.Message):
    """Response message for ScheduleService.ListSchedules

  Fields:
    nextPageToken: A token to retrieve the next page of results. Pass to
      ListSchedulesRequest.page_token to obtain that page.
    schedules: List of Schedules in the requested page.
  """
    nextPageToken = _messages.StringField(1)
    schedules = _messages.MessageField('GoogleCloudAiplatformV1beta1Schedule', 2, repeated=True)