from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3alphaConditionContext(_messages.Message):
    """Additional context for troubleshooting conditional role bindings and
  deny rules.

  Fields:
    destination: The destination of a network activity, such as accepting a
      TCP connection. In a multi-hop network activity, the destination
      represents the receiver of the last hop.
    effectiveTags: Output only. The effective tags on the resource. The
      effective tags are fetched during troubleshooting.
    request: Represents a network request, such as an HTTP request.
    resource: Represents a target resource that is involved with a network
      activity. If multiple resources are involved with an activity, this must
      be the primary one.
  """
    destination = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaConditionContextPeer', 1)
    effectiveTags = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaConditionContextEffectiveTag', 2, repeated=True)
    request = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaConditionContextRequest', 3)
    resource = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaConditionContextResource', 4)