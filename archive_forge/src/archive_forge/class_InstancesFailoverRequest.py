from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class InstancesFailoverRequest(_messages.Message):
    """Instance failover request.

  Fields:
    failoverContext: Failover Context.
  """
    failoverContext = _messages.MessageField('FailoverContext', 1)