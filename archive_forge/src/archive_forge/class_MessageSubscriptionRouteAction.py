from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MessageSubscriptionRouteAction(_messages.Message):
    """Specifies the action to take for a route.

  Fields:
    destination: Required. The destination to deliver the message to.
  """
    destination = _messages.MessageField('MessageSubscriptionDestination', 1)