from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotifierSpec(_messages.Message):
    """NotifierSpec is the configuration container for notifications.

  Fields:
    notification: The configuration of this particular notifier.
    secrets: Configurations for secret resources used by this particular
      notifier.
  """
    notification = _messages.MessageField('Notification', 1)
    secrets = _messages.MessageField('NotifierSecret', 2, repeated=True)