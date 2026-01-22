from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityOrganizationsLocationsSecurityProfilesPatchRequest(_messages.Message):
    """A NetworksecurityOrganizationsLocationsSecurityProfilesPatchRequest
  object.

  Fields:
    name: Immutable. Identifier. Name of the SecurityProfile resource. It
      matches pattern `projects|organizations/*/locations/{location}/securityP
      rofiles/{security_profile}`.
    securityProfile: A SecurityProfile resource to be passed as the request
      body.
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the SecurityProfile resource by the update. The fields
      specified in the update_mask are relative to the resource, not the full
      request. A field will be overwritten if it is in the mask.
  """
    name = _messages.StringField(1, required=True)
    securityProfile = _messages.MessageField('SecurityProfile', 2)
    updateMask = _messages.StringField(3)