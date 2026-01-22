from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListSchedulesResponse(_messages.Message):
    """Response for listing scheduled notebook job.

  Fields:
    nextPageToken: Page token that can be used to continue listing from the
      last result in the next list call.
    schedules: A list of returned instances.
    unreachable: Schedules that could not be reached. For example:
      ['projects/{project_id}/location/{location}/schedules/monthly_digest',
      'projects/{project_id}/location/{location}/schedules/weekly_sentiment']
  """
    nextPageToken = _messages.StringField(1)
    schedules = _messages.MessageField('Schedule', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)