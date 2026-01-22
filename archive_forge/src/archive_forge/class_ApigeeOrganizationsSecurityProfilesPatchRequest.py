from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSecurityProfilesPatchRequest(_messages.Message):
    """A ApigeeOrganizationsSecurityProfilesPatchRequest object.

  Fields:
    googleCloudApigeeV1SecurityProfile: A GoogleCloudApigeeV1SecurityProfile
      resource to be passed as the request body.
    name: Immutable. Name of the security profile resource. Format:
      organizations/{org}/securityProfiles/{profile}
    updateMask: Required. The list of fields to update.
  """
    googleCloudApigeeV1SecurityProfile = _messages.MessageField('GoogleCloudApigeeV1SecurityProfile', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)