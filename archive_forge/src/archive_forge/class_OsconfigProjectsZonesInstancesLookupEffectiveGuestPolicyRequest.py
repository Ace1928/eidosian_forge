from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsZonesInstancesLookupEffectiveGuestPolicyRequest(_messages.Message):
    """A OsconfigProjectsZonesInstancesLookupEffectiveGuestPolicyRequest
  object.

  Fields:
    instance: Required. The VM instance whose policies are being looked up.
    lookupEffectiveGuestPolicyRequest: A LookupEffectiveGuestPolicyRequest
      resource to be passed as the request body.
  """
    instance = _messages.StringField(1, required=True)
    lookupEffectiveGuestPolicyRequest = _messages.MessageField('LookupEffectiveGuestPolicyRequest', 2)