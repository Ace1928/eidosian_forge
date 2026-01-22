from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterV2alpha1ConditionContext(_messages.Message):
    """Represents the attributes that will be used to do IAM condition
  evaluation.

  Fields:
    destination: The destination of a network activity, such as accepting a
      TCP connection. In a multi hop network activity, the destination
      represents the receiver of the last hop.
    request: Represents a network request, such as an HTTP request.
    resource: Represents a target resource that is involved with a network
      activity. If multiple resources are involved with an activity, this must
      be the primary one.
  """
    destination = _messages.MessageField('GoogleCloudPolicytroubleshooterV2alpha1Peer', 1)
    request = _messages.MessageField('GoogleCloudPolicytroubleshooterV2alpha1Request', 2)
    resource = _messages.MessageField('GoogleCloudPolicytroubleshooterV2alpha1Resource', 3)