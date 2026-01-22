from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsSchedulesTriggerRequest(_messages.Message):
    """A NotebooksProjectsLocationsSchedulesTriggerRequest object.

  Fields:
    name: Required. Format: `parent=projects/{project_id}/locations/{location}
      /schedules/{schedule_id}`
    triggerScheduleRequest: A TriggerScheduleRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    triggerScheduleRequest = _messages.MessageField('TriggerScheduleRequest', 2)