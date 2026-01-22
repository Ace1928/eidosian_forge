from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsNotificationChannelsDeleteRequest(_messages.Message):
    """A MonitoringProjectsNotificationChannelsDeleteRequest object.

  Fields:
    force: If true, the notification channel will be deleted regardless of its
      use in alert policies (the policies will be updated to remove the
      channel). If false, channels that are still referenced by an existing
      alerting policy will fail to be deleted in a delete operation.
    name: Required. The channel for which to execute the request. The format
      is: projects/[PROJECT_ID_OR_NUMBER]/notificationChannels/[CHANNEL_ID]
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)