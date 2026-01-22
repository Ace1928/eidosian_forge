from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RRSetRoutingPolicyWrrPolicy(_messages.Message):
    """Configures a RRSetRoutingPolicy that routes in a weighted round robin
  fashion.

  Fields:
    items: A RRSetRoutingPolicyWrrPolicyWrrPolicyItem attribute.
    kind: A string attribute.
  """
    items = _messages.MessageField('RRSetRoutingPolicyWrrPolicyWrrPolicyItem', 1, repeated=True)
    kind = _messages.StringField(2, default='dns#rRSetRoutingPolicyWrrPolicy')