from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RoutersGetRoutePolicyResponse(_messages.Message):
    """A RoutersGetRoutePolicyResponse object.

  Fields:
    resource: A RoutePolicy attribute.
  """
    resource = _messages.MessageField('RoutePolicy', 1)