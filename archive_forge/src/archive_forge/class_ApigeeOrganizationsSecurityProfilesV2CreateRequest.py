from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSecurityProfilesV2CreateRequest(_messages.Message):
    """A ApigeeOrganizationsSecurityProfilesV2CreateRequest object.

  Fields:
    googleCloudApigeeV1SecurityProfileV2: A
      GoogleCloudApigeeV1SecurityProfileV2 resource to be passed as the
      request body.
    parent: Required. The parent resource name.
    securityProfileV2Id: Required. The security profile id.
  """
    googleCloudApigeeV1SecurityProfileV2 = _messages.MessageField('GoogleCloudApigeeV1SecurityProfileV2', 1)
    parent = _messages.StringField(2, required=True)
    securityProfileV2Id = _messages.StringField(3)