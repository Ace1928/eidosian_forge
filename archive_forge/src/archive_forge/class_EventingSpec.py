from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EventingSpec(_messages.Message):
    """EventingSpec defines the desired state of Eventing

  Fields:
    enabled: A boolean attribute.
  """
    enabled = _messages.BooleanField(1)