from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV1p1beta1NotificationMessage(_messages.Message):
    """Security Command Center's Notification

  Fields:
    finding: If it's a Finding based notification config, this field will be
      populated.
    notificationConfigName: Name of the notification config that generated
      current notification.
    resource: The Cloud resource tied to the notification.
  """
    finding = _messages.MessageField('GoogleCloudSecuritycenterV1p1beta1Finding', 1)
    notificationConfigName = _messages.StringField(2)
    resource = _messages.MessageField('GoogleCloudSecuritycenterV1p1beta1Resource', 3)