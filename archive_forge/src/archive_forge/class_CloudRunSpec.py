from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudRunSpec(_messages.Message):
    """CloudRunSpec defines the desired state of CloudRun

  Fields:
    eventing: `json:"eventing,omitempty"`
    networking: `json:"networking,omitempty"`
    serving: `json:"serving,omitempty"`
  """
    eventing = _messages.MessageField('EventingSpec', 1)
    networking = _messages.MessageField('NetworkingSpec', 2)
    serving = _messages.MessageField('ServingSpec', 3)